using System.Xml.Linq;
using Microsoft.ML.OnnxRuntime;
using Windows.Management.Deployment;

namespace WinMLResNet;

/// <summary>
/// Information about one execution provider discovered from an installed MSIX package.
/// </summary>
public class EpRegistrationInfo
{
    public string Name { get; set; } = "";
    public string PackageFamily { get; set; } = "";
    public string LibraryPath { get; set; } = "";
    public bool Registered { get; set; }
    public string? Error { get; set; }

    public string Summary =>
        $"{Name}: {(Registered ? "registered" : "not registered")}" +
        (string.IsNullOrEmpty(Error) ? "" : $" ({Error})");
}

/// <summary>
/// Re-implements the Windows ML execution-provider discovery + registration mechanism from scratch,
/// following the runtime's own <c>ExecutionProviderStore.cpp</c> and the <c>winml</c> CLI:
///
/// <list type="number">
///   <item><b>Enumerate package extensions</b> — every WinML EP MSIX declares a
///     <c>com.microsoft.windowsmlruntime.executionprovider[.2/.3]</c> package extension whose
///     properties carry the provider registration name and DLL name.</item>
///   <item><b>Add each EP package to the process package graph</b> via
///     <c>TryCreatePackageDependency</c> + <c>AddPackageDependency</c> (see
///     <see cref="PackageDependency"/>). This resolves the EP DLL's dependency closure — most
///     importantly sibling DLLs such as <c>onnxruntime_providers_shared.dll</c> that live in the
///     WinML runtime framework package — onto the loader search path.</item>
///   <item><b>Register the provider library with ONNX Runtime</b> via
///     <see cref="OrtEnv.RegisterExecutionProviderLibrary(string, string)"/>, pointing at
///     <c>&lt;InstalledLocation&gt;\&lt;PublicFolder&gt;\&lt;ProviderPath&gt;</c>.</item>
/// </list>
///
/// <para>This is intentionally lower-level than <c>ExecutionProviderCatalog.TryRegister()</c>: it
/// gives explicit control over the package-graph step that makes vendor EP libraries loadable.
/// Providers register regardless of certification; whether a provider then exposes a selectable
/// device is decided by the EP itself in <c>OrtEnv.GetEpDevices()</c> based on present hardware.
/// The built-in CPU and DirectML providers need no registration.</para>
/// </summary>
public static class EpRegistration
{
    // WinML EP package extensions, newest contract first (V3 supports multiple providers per package).
    private static readonly string[] s_epExtensionNames =
    {
        "com.microsoft.windowsmlruntime.executionprovider.3",
        "com.microsoft.windowsmlruntime.executionprovider.2",
        "com.microsoft.windowsmlruntime.executionprovider",
    };

    // WinML EP authoring convention: PublicFolder="ExecutionProvider".
    private const string EpPublicFolder = "ExecutionProvider";

    // Registration is process-global in ORT, so guard against re-registering the same provider.
    private static readonly HashSet<string> s_registered = new(StringComparer.OrdinalIgnoreCase);
    private static readonly object s_lock = new();

    /// <summary>Results of the most recent full <see cref="RegisterAllAsync"/> pass, for UI diagnostics.</summary>
    public static IReadOnlyList<EpRegistrationInfo> LastResults { get; private set; } = Array.Empty<EpRegistrationInfo>();

    public static Task<List<EpRegistrationInfo>> RegisterAllAsync() => Task.Run(() => Register(null));

    /// <summary>
    /// Registers only the named providers in-process. Used by the app to register the known-safe
    /// set (providers a crash-isolated worker confirmed produce a device), so the UI process never
    /// tries to load a fragile vendor library that could native-crash on hardware without the
    /// matching accelerator.
    /// </summary>
    public static Task RegisterSpecificAsync(IEnumerable<string> providerNames)
    {
        var set = new HashSet<string>(providerNames, StringComparer.OrdinalIgnoreCase);
        return Task.Run(() => Register(set));
    }

    private record EpCandidate(string ProviderName, string LibraryPath, string PackageFamily, ulong Version, int ContractRank);

    private static List<EpRegistrationInfo> Register(HashSet<string>? onlyProviders)
    {
        var ortEnv = OrtEnv.Instance();
        var results = new List<EpRegistrationInfo>();

        List<EpCandidate> candidates;
        try
        {
            candidates = DiscoverProviders();
        }
        catch (Exception ex)
        {
            var fail = new List<EpRegistrationInfo> { new() { Name = "(discovery)", Error = ex.Message } };
            if (onlyProviders is null) LastResults = fail;
            return fail;
        }

        foreach (var cand in candidates)
        {
            if (onlyProviders is not null && !onlyProviders.Contains(cand.ProviderName)) continue;

            var info = new EpRegistrationInfo
            {
                Name = cand.ProviderName,
                PackageFamily = cand.PackageFamily,
                LibraryPath = cand.LibraryPath,
            };

            lock (s_lock)
            {
                if (s_registered.Contains(cand.ProviderName))
                {
                    info.Registered = true;
                    results.Add(info);
                    continue;
                }
            }

            // Step 2: bring the EP package (and its declared dependency closure) into this process's
            // package graph so the provider DLL's sibling dependencies resolve on the loader path.
            if (!PackageDependency.EnsureInProcessGraph(cand.PackageFamily, out var depError))
            {
                // Non-fatal: registration may still work if the DLL is self-contained. Record it.
                info.Error = depError is null ? null : $"AddPackageDependency: {depError}";
            }

            // Step 2b: add the provider DLL's own folder to the loader search path. Some vendor EPs
            // ship their private runtime DLLs (e.g. cudart64_12.dll for NVIDIA TensorRT, ryzen_mm.dll
            // for AMD Ryzen) alongside the provider DLL; ORT's default search dirs don't include the
            // DLL's own directory, so those siblings must be made discoverable explicitly.
            PackageDependency.AddSearchDirectory(Path.GetDirectoryName(cand.LibraryPath));

            // Step 3: register the provider library with ONNX Runtime.
            try
            {
                if (!File.Exists(cand.LibraryPath))
                {
                    info.Error = AppendError(info.Error, "provider library not found");
                }
                else
                {
                    ortEnv.RegisterExecutionProviderLibrary(cand.ProviderName, cand.LibraryPath);
                    info.Registered = true;
                    lock (s_lock) { s_registered.Add(cand.ProviderName); }
                }
            }
            catch (Exception ex)
            {
                info.Error = AppendError(info.Error, ex.Message.Replace("\r", " ").Replace("\n", " "));
            }

            results.Add(info);
        }

        if (onlyProviders is null) LastResults = results;
        return results;
    }

    /// <summary>
    /// Step 1: enumerate installed packages and read their WinML EP package extensions to produce
    /// (providerName, libraryPath, packageFamily) candidates, deduped by (family, provider) keeping
    /// the highest package version.
    /// </summary>
    private static List<EpCandidate> DiscoverProviders()
    {
        var pm = new PackageManager();
        var packages = pm.FindPackagesForUser(string.Empty);

        // Dedup by (family|provider), keep highest version.
        var best = new Dictionary<string, EpCandidate>(StringComparer.OrdinalIgnoreCase);

        foreach (var pkg in packages)
        {
            string family;
            ulong version;
            string installPath;
            try
            {
                family = pkg.Id.FamilyName;
                var v = pkg.Id.Version;
                version = ((ulong)v.Major << 48) | ((ulong)v.Minor << 32) | ((ulong)v.Build << 16) | v.Revision;
                installPath = pkg.InstalledLocation.Path;
            }
            catch { continue; }

            if (!LooksLikeEpPackage(family)) continue;

            var manifestPath = Path.Combine(installPath, "AppxManifest.xml");
            if (!File.Exists(manifestPath)) continue;

            foreach (var (providerName, publicFolder, providerPath, contractRank) in ParseEpExtensions(manifestPath))
            {
                // libraryPath = <InstalledLocation>\<PublicFolder>\<ProviderPath>, with the WinML
                // authoring-convention fallback of "ExecutionProvider" when the folder is absent.
                var folder = string.IsNullOrEmpty(publicFolder) ? EpPublicFolder : publicFolder;
                var libraryPath = Path.Combine(installPath, folder, providerPath);

                // Dedup by (family, provider). A single package can declare the same provider under
                // multiple extension contracts (e.g. VitisAI under v1 → onnxruntime_providers_vitisai.dll
                // and v2 → onnxruntime_vitisai_ep.dll); the runtime prefers the newest contract, which
                // points at the self-contained DLL. So keep the higher (package version, contract rank).
                var key = $"{family}|{providerName}";
                if (best.TryGetValue(key, out var existing) &&
                    (existing.Version > version ||
                     (existing.Version == version && existing.ContractRank >= contractRank)))
                {
                    continue;
                }
                best[key] = new EpCandidate(providerName, libraryPath, family, version, contractRank);
            }
        }

        return best.Values.ToList();
    }

    private static bool LooksLikeEpPackage(string family) =>
        family.Contains("WinML", StringComparison.OrdinalIgnoreCase) ||
        family.Contains(".EP.", StringComparison.OrdinalIgnoreCase) ||
        family.Contains("ExecutionProvider", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Reads the WinML EP package extensions from an AppxManifest and yields
    /// (providerName, publicFolder, providerPath, contractRank) for each. Matches the extension by
    /// the <c>com.microsoft.windowsmlruntime.executionprovider[.N]</c> names (v1/v2/v3); contractRank
    /// is 3/2/1 so callers can prefer the newest contract.
    /// </summary>
    private static IEnumerable<(string ProviderName, string PublicFolder, string ProviderPath, int ContractRank)> ParseEpExtensions(string manifestPath)
    {
        XDocument doc;
        try { doc = XDocument.Load(manifestPath); }
        catch { yield break; }

        foreach (var ext in doc.Descendants().Where(e => e.Name.LocalName == "PackageExtension"))
        {
            var name = (string?)ext.Attribute("Name");
            if (name is null) continue;
            int rank = Array.IndexOf(s_epExtensionNames, name) switch
            {
                0 => 3,   // ...executionprovider.3
                1 => 2,   // ...executionprovider.2
                2 => 1,   // ...executionprovider
                _ => 0,
            };
            if (rank == 0) continue;

            var publicFolder = (string?)ext.Attribute("PublicFolder") ?? "";
            var props = ext.Descendants().FirstOrDefault(e => e.Name.LocalName == "Properties");
            var providerName = props?.Descendants().FirstOrDefault(e => e.Name.LocalName == "ProviderName")?.Value;
            var providerPath = props?.Descendants().FirstOrDefault(e => e.Name.LocalName == "ProviderPath")?.Value;

            if (!string.IsNullOrEmpty(providerName) && !string.IsNullOrEmpty(providerPath))
                yield return (providerName!, publicFolder, providerPath!, rank);
        }
    }

    private static string AppendError(string? existing, string add) =>
        string.IsNullOrEmpty(existing) ? add : $"{existing}; {add}";
}
