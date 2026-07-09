using System.Diagnostics;
using System.Text.Json;

namespace WinMLResNet;

/// <summary>An EP device reported by the isolated inventory worker.</summary>
public class EpInventoryDevice
{
    public string Ep { get; set; } = "";
    public string DeviceType { get; set; } = "";
    public string Vendor { get; set; } = "";
}

/// <summary>Per-provider registration diagnostic reported by the isolated inventory worker.</summary>
public class EpInventoryDiag
{
    public string Provider { get; set; } = "";
    public bool Registered { get; set; }
    public string PackageFamily { get; set; } = "";
    public string? Error { get; set; }
}

/// <summary>
/// Runs the isolated worker's <c>--list-eps</c> mode to discover which execution providers are
/// available, without force-loading fragile vendor EP libraries in the UI process.
///
/// <para>Force-registering every installed EP in-process is unsafe: a vendor EP library that probes
/// for absent hardware (e.g. MIGraphX calling into ROCm) can <c>__fastfail</c> the process, which a
/// managed try/catch cannot recover from. By doing the registration + enumeration in a child
/// process, that risk is contained; the app then registers in-process only the providers that were
/// confirmed to produce a usable device.</para>
/// </summary>
public static class EpInventory
{
    private class WorkerJson
    {
        public bool ok { get; set; }
        public string? error { get; set; }
        public List<Dev> devices { get; set; } = new();
        public List<Diag> diagnostics { get; set; } = new();
        public class Dev { public string? ep { get; set; } public string? deviceType { get; set; } public string? vendor { get; set; } }
        public class Diag { public string? provider { get; set; } public bool registered { get; set; } public string? packageFamily { get; set; } public string? error { get; set; } }
    }

    public static IReadOnlyList<EpInventoryDevice> Devices { get; private set; } = Array.Empty<EpInventoryDevice>();
    public static IReadOnlyList<EpInventoryDiag> Diagnostics { get; private set; } = Array.Empty<EpInventoryDiag>();
    public static bool Loaded { get; private set; }
    public static string? Error { get; private set; }

    /// <summary>Provider names that enumerated at least one device — the set that is safe to
    /// register in-process. Always includes the built-in CPU/DML providers.</summary>
    public static IReadOnlyCollection<string> DeviceProducingProviders =>
        Devices.Select(d => d.Ep).Distinct(StringComparer.OrdinalIgnoreCase).ToList();

    /// <summary>
    /// Device kinds each vendor EP can target, from the Windows ML EP catalog
    /// (ExecutionProviderCatalog.json "SupportedDeviceKinds"). Used to offer explicit compile
    /// targets (e.g. "OpenVINO CPU") even when the provider hasn't enumerated a device on the
    /// current hardware — selecting one attempts it and reports honestly if the device is absent.
    /// </summary>
    private static readonly Dictionary<string, string[]> s_catalogDeviceKinds = new(StringComparer.OrdinalIgnoreCase)
    {
        ["OpenVINOExecutionProvider"] = new[] { "CPU", "GPU", "NPU" },
        ["NvTensorRTRTXExecutionProvider"] = new[] { "GPU" },
        ["QNNExecutionProvider"] = new[] { "GPU", "NPU" },
        ["VitisAIExecutionProvider"] = new[] { "NPU" },
        ["MIGraphXExecutionProvider"] = new[] { "GPU" },
        ["WinMLCGExecutionProvider"] = new[] { "GPU" },
    };

    /// <summary>A vendor EP + device-type the user can attempt to compile on.</summary>
    public record AttemptTarget(string Ep, string DeviceType);

    /// <summary>
    /// Registered vendor providers that did not enumerate a device on the current hardware,
    /// expanded to their catalog device kinds. On matching hardware these enumerate for real; here
    /// they let the user explicitly request e.g. "OpenVINO CPU" and see the honest result.
    /// </summary>
    public static IEnumerable<AttemptTarget> RegisteredButUnenumeratedTargets()
    {
        var enumerated = new HashSet<string>(Devices.Select(d => d.Ep), StringComparer.OrdinalIgnoreCase);
        foreach (var diag in Diagnostics)
        {
            if (!diag.Registered || enumerated.Contains(diag.Provider)) continue;
            if (!s_catalogDeviceKinds.TryGetValue(diag.Provider, out var kinds)) continue;
            foreach (var kind in kinds)
                yield return new AttemptTarget(diag.Provider, kind);
        }
    }

    private static Task<bool>? s_ensureTask;
    private static readonly object s_lock = new();

    /// <summary>Runs the inventory worker once and caches the result.</summary>
    public static Task<bool> EnsureAsync()
    {
        lock (s_lock)
        {
            return s_ensureTask ??= Task.Run(RunWorker);
        }
    }

    private static bool RunWorker()
    {
        var worker = CompileBenchmarkRunner.LocateWorker();
        if (worker is null)
        {
            Error = "Compile worker not found — build the CompileErrorTest project.";
            return false;
        }

        var outFile = Path.Combine(Path.GetTempPath(), $"ep-inventory-{Guid.NewGuid():N}.json");
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = worker,
                WorkingDirectory = Path.GetDirectoryName(worker),
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };
            psi.ArgumentList.Add("--list-eps");
            psi.ArgumentList.Add("--out");
            psi.ArgumentList.Add(outFile);

            using var proc = Process.Start(psi);
            if (proc is null) { Error = "Failed to start inventory worker."; return false; }

            _ = proc.StandardOutput.ReadToEndAsync();
            _ = proc.StandardError.ReadToEndAsync();
            if (!proc.WaitForExit(120000))
            {
                try { proc.Kill(true); } catch { }
                Error = "Inventory worker timed out.";
                return false;
            }

            if (!File.Exists(outFile))
            {
                Error = $"Inventory worker exited (0x{proc.ExitCode & 0xFFFFFFFF:X8}) without a result.";
                return false;
            }

            var dto = JsonSerializer.Deserialize<WorkerJson>(
                File.ReadAllText(outFile), new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            if (dto is null) { Error = "Empty inventory result."; return false; }

            Devices = dto.devices.Select(d => new EpInventoryDevice
            {
                Ep = d.ep ?? "",
                DeviceType = d.deviceType ?? "",
                Vendor = d.vendor ?? ""
            }).ToList();

            Diagnostics = dto.diagnostics.Select(d => new EpInventoryDiag
            {
                Provider = d.provider ?? "",
                Registered = d.registered,
                PackageFamily = d.packageFamily ?? "",
                Error = d.error
            }).ToList();

            Error = dto.ok ? null : dto.error;
            Loaded = true;
            return true;
        }
        catch (Exception ex)
        {
            Error = ex.Message;
            return false;
        }
        finally
        {
            try { if (File.Exists(outFile)) File.Delete(outFile); } catch { }
        }
    }
}
