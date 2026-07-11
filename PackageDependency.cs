using System.Runtime.InteropServices;

namespace WinMLResNet;

/// <summary>
/// Adds an installed MSIX package (by package family name) into the current process's dynamic
/// dependency package graph, using the Win32 <c>TryCreatePackageDependency</c> +
/// <c>AddPackageDependency</c> APIs.
///
/// <para>Why this is needed for execution providers: a Windows ML vendor EP DLL (e.g.
/// <c>onnxruntime_providers_vitisai.dll</c>) depends on sibling DLLs (e.g.
/// <c>onnxruntime_providers_shared.dll</c>) that ship in a *different* package (the WinML runtime
/// framework). Registering the EP DLL by raw path fails with "depends on X.dll which is missing"
/// because those directories aren't on the loader search path. Bringing the EP package into the
/// process package graph resolves its declared dependency closure so those DLLs load.</para>
///
/// <para>This mirrors the Windows ML runtime's own mechanism (ExecutionProviderStore.cpp /
/// RuntimeDependencyManagement.h) and the winml CLI's <c>TryAddNativePackageDependency</c>.
/// Requires Windows 11 21H2+ (build 22000). Uses
/// <c>PackageDependencyLifetimeKind_Process</c>, so the OS auto-releases on process exit — no
/// explicit delete is required.</para>
/// </summary>
public static class PackageDependency
{
    // --- Win32 enums (appmodel.h) ---
    [Flags]
    private enum ProcessorArchitectures : uint
    {
        None = 0x0,
        Neutral = 0x1,
        X86 = 0x2,
        X64 = 0x4,
        Arm = 0x8,
        Arm64 = 0x10,
        X86OnArm64 = 0x20,
    }

    private enum LifetimeKind : uint
    {
        Process = 0,
        FilePath = 1,
        RegistryKey = 2,
    }

    [Flags]
    private enum CreateOptions : uint
    {
        None = 0x0,
        DoNotVerifyDependencyResolution = 0x1,
        ScopeIsSystem = 0x2,
    }

    [Flags]
    private enum AddOptions : uint
    {
        None = 0x0,
        PrependIfRankCollision = 0x1,
    }

    // PACKAGE_VERSION is a union that is effectively a UINT64; 0 means "any version".
    [DllImport("kernelbase.dll", CharSet = CharSet.Unicode, ExactSpelling = true)]
    private static extern int TryCreatePackageDependency(
        IntPtr user,
        string packageFamilyName,
        ulong minVersion,
        ProcessorArchitectures architectures,
        LifetimeKind lifetimeKind,
        string? lifetimeArtifact,
        CreateOptions options,
        out IntPtr packageDependencyId);

    [DllImport("kernelbase.dll", CharSet = CharSet.Unicode, ExactSpelling = true)]
    private static extern int AddPackageDependency(
        IntPtr packageDependencyId,
        int rank,
        AddOptions options,
        out IntPtr packageDependencyContext,
        out IntPtr packageFullName);

    [DllImport("kernel32.dll")]
    private static extern IntPtr GetProcessHeap();

    [DllImport("kernel32.dll")]
    private static extern bool HeapFree(IntPtr heap, uint flags, IntPtr mem);

    // Adds a directory to the process DLL search path (contributes to LOAD_LIBRARY_SEARCH_USER_DIRS,
    // itself part of LOAD_LIBRARY_SEARCH_DEFAULT_DIRS). Returns an opaque cookie or NULL on failure.
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr AddDllDirectory(string newDirectory);

    private static readonly HashSet<string> s_dllDirs = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Adds <paramref name="directory"/> to the process DLL search path so a vendor EP DLL's sibling
    /// dependencies that live in the *same* folder (e.g. <c>cudart64_12.dll</c> next to the NVIDIA
    /// TensorRT EP, or <c>ryzen_mm.dll</c> next to the AMD Ryzen EP) resolve when ONNX Runtime loads
    /// the provider library. Mirrors the runtime's use of <c>LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR</c>.
    /// Idempotent per directory; failures are non-fatal.
    /// </summary>
    public static void AddSearchDirectory(string? directory)
    {
        if (string.IsNullOrWhiteSpace(directory) || !Directory.Exists(directory)) return;
        var full = Path.GetFullPath(directory);
        lock (s_lock)
        {
            if (!s_dllDirs.Add(full)) return;
        }
        try { AddDllDirectory(full); } catch { /* non-fatal */ }
    }

    private static ProcessorArchitectures CurrentArchitecture => RuntimeInformation.ProcessArchitecture switch
    {
        Architecture.Arm64 => ProcessorArchitectures.Arm64,
        Architecture.X86 => ProcessorArchitectures.X86,
        _ => ProcessorArchitectures.X64,
    };

    // Families we've already added this process — the graph entry persists for the process
    // lifetime, so adding twice is wasteful (and would stack redundant entries).
    private static readonly HashSet<string> s_added = new(StringComparer.OrdinalIgnoreCase);
    private static readonly object s_lock = new();

    /// <summary>
    /// Ensures the given package family is present in this process's package graph. Returns true on
    /// success (or if already added). On failure, <paramref name="error"/> carries the HRESULT.
    /// </summary>
    public static bool EnsureInProcessGraph(string packageFamilyName, out string? error)
    {
        error = null;
        if (string.IsNullOrWhiteSpace(packageFamilyName)) { error = "empty family name"; return false; }

        lock (s_lock)
        {
            if (s_added.Contains(packageFamilyName)) return true;
        }

        IntPtr dependencyId = IntPtr.Zero;
        try
        {
            int hr = TryCreatePackageDependency(
                IntPtr.Zero,                                  // current user
                packageFamilyName,
                0UL,                                          // any version
                CurrentArchitecture,
                LifetimeKind.Process,
                null,                                         // no lifetime artifact
                CreateOptions.DoNotVerifyDependencyResolution,
                out dependencyId);
            if (hr < 0)
            {
                error = $"TryCreatePackageDependency 0x{hr:X8}";
                return false;
            }

            hr = AddPackageDependency(
                dependencyId,
                0,                                            // rank
                AddOptions.None,
                out _,                                        // context (Process lifetime → auto-freed)
                out IntPtr packageFullName);
            if (packageFullName != IntPtr.Zero)
                HeapFree(GetProcessHeap(), 0, packageFullName);

            if (hr < 0)
            {
                error = $"AddPackageDependency 0x{hr:X8}";
                return false;
            }

            lock (s_lock) { s_added.Add(packageFamilyName); }
            return true;
        }
        catch (Exception ex)
        {
            error = ex.Message;
            return false;
        }
        finally
        {
            if (dependencyId != IntPtr.Zero)
                HeapFree(GetProcessHeap(), 0, dependencyId);
        }
    }
}
