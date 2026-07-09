using System.Diagnostics;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.Windows.AI.MachineLearning;
using WinMLResNet;

// WinML EP worker
// ---------------
// Modes:
//
//  1. Compile job (isolated worker used by the WinUI app):
//       CompileErrorTest --compile --model <path> --ep <EpName> --deviceType <CPU|GPU|NPU>
//                        --out <resultJson> [--label <text>]
//     Registers all installed EPs, selects the requested EP device, times
//     OrtModelCompilationOptions.CompileModel(), and writes a JSON result file.
//     Running this in a child process isolates native crashes (e.g. a GPU EP on a
//     machine with no compatible GPU) so they can't take down the UI.
//
//  2. List EPs (isolated inventory used by the WinUI app):
//       CompileErrorTest --list-eps --out <json>
//     Registers all installed EPs and writes the enumerated devices + per-provider
//     registration diagnostics as JSON. Runs out-of-process so force-loading a
//     fragile vendor EP library can't crash the app.
//
//  3. Probe (headless diagnostic):
//       CompileErrorTest [modelPath] [--all] [--compile-only]
//     Prints which EPs the runtime enumerates and times JIT + AOT compile per EP.

if (args.Contains("--compile"))
{
    await RunCompileJobAsync(args);
    return;
}

if (args.Contains("--list-eps"))
{
    await RunListEpsAsync(args);
    return;
}

if (args.Contains("--cpu-plain"))
{
    RunCpuPlain(args);
    return;
}

await RunProbeAsync(args);


// ----------------------------------------------------------------------------
// CPU-plain test: compile using ONLY the built-in ORT CPU EP.
// No EP registration, no ExecutionProviderCatalog, no GetEpDevices, no
// AppendExecutionProvider — a default SessionOptions already includes the CPU
// provider that is statically linked into onnxruntime.dll.
// ----------------------------------------------------------------------------
static void RunCpuPlain(string[] args)
{
    var model = GetArg(args, "--model") ?? ResolveDefaultModel();
    Console.WriteLine("=== CPU-plain compile (built-in ORT CPU EP, no registration) ===");
    Console.WriteLine($"Model: {model}");

    var envOptions = new EnvironmentCreationOptions
    {
        logId = "CpuPlain",
        logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
    };
    OrtEnv.CreateInstanceWithOptions(ref envOptions);

    // NOTE: no Ep registration and no provider is appended. The CPU EP is the
    // implicit default/fallback provider in every SessionOptions.
    var so = new SessionOptions();
    var compiledPath = Path.Combine(Path.GetTempPath(), $"cpuplain-{Path.GetFileNameWithoutExtension(model)}.onnx");
    if (File.Exists(compiledPath)) File.Delete(compiledPath);

    var sw = Stopwatch.StartNew();
    using (var compileOptions = new OrtModelCompilationOptions(so))
    {
        compileOptions.SetInputModelPath(model);
        compileOptions.SetOutputModelPath(compiledPath);
        compileOptions.CompileModel();
    }
    sw.Stop();

    long bytes = File.Exists(compiledPath) ? new FileInfo(compiledPath).Length : -1;
    Console.WriteLine($"Compiled OK in {sw.Elapsed.TotalMilliseconds:F1} ms ({bytes:N0} bytes) — no EP registration used.");
    try { if (File.Exists(compiledPath)) File.Delete(compiledPath); } catch { }
}


// ----------------------------------------------------------------------------
// List-EPs mode (isolated inventory)
// ----------------------------------------------------------------------------
static async Task RunListEpsAsync(string[] args)
{
    string outPath = GetArg(args, "--out") ?? Path.Combine(Path.GetTempPath(), "ep-inventory.json");
    var inventory = new EpInventoryResult();

    try
    {
        var envOptions = new EnvironmentCreationOptions
        {
            logId = "EpInventory",
            logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
        };
        OrtEnv.CreateInstanceWithOptions(ref envOptions);
        var ortEnv = OrtEnv.Instance();

        var infos = await EpRegistration.RegisterAllAsync();
        foreach (var i in infos)
            inventory.diagnostics.Add(new EpDiag(i.Name, i.Registered, i.PackageFamily, i.Error));

        foreach (var d in ortEnv.GetEpDevices())
        {
            string vendor = "";
            try { vendor = d.EpVendor; } catch { }
            inventory.devices.Add(new EpDev(d.EpName, d.HardwareDevice.Type.ToString(), vendor));
        }
        inventory.ok = true;
    }
    catch (Exception ex)
    {
        inventory.ok = false;
        inventory.error = $"0x{ex.HResult:X8} {ex.Message}".Replace("\r", " ").Replace("\n", " ");
    }

    try { File.WriteAllText(outPath, JsonSerializer.Serialize(inventory)); }
    catch { /* parent treats a missing file as a crash */ }
}


// ----------------------------------------------------------------------------
// Compile job mode
// ----------------------------------------------------------------------------
static async Task RunCompileJobAsync(string[] args)
{
    string model = GetArg(args, "--model") ?? "";
    string ep = GetArg(args, "--ep") ?? "";
    string deviceType = GetArg(args, "--deviceType") ?? "";
    string outPath = GetArg(args, "--out") ?? Path.Combine(Path.GetTempPath(), "compile-result.json");
    string label = GetArg(args, "--label") ?? Path.GetFileName(model);

    double compileMs = -1;
    long compiledBytes = -1;
    bool ok = false;
    string? error = null;

    try
    {
        if (!File.Exists(model))
            throw new FileNotFoundException($"Model not found: {model}");

        var envOptions = new EnvironmentCreationOptions
        {
            logId = "EpCompileWorker",
            logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
        };
        OrtEnv.CreateInstanceWithOptions(ref envOptions);
        var ortEnv = OrtEnv.Instance();

        await EpRegistration.RegisterAllAsync();

        var devices = ortEnv.GetEpDevices();
        OrtEpDevice? target = null;
        foreach (var d in devices)
        {
            if (!string.Equals(d.EpName, ep, StringComparison.OrdinalIgnoreCase)) continue;
            if (!string.IsNullOrEmpty(deviceType) &&
                !string.Equals(d.HardwareDevice.Type.ToString(), deviceType, StringComparison.OrdinalIgnoreCase))
                continue;
            target = d;
            break;
        }

        if (target is null)
            throw new InvalidOperationException(
                $"{ep} did not enumerate a {deviceType} device on this system — the provider registered but no matching hardware was found. " +
                $"({devices.Count} device(s) available: {string.Join(", ", devices.Select(d => $"{d.EpName}/{d.HardwareDevice.Type}"))})");

        var so = new SessionOptions();
        so.AppendExecutionProvider(ortEnv, new[] { target }, EpOptionsFor(target.EpName));

        var epSafe = ep.Replace(" ", "_").Replace(".", "_").Replace("(", "").Replace(")", "");
        var compiledPath = Path.Combine(
            Path.GetTempPath(),
            $"compile-{epSafe}-{deviceType}-{Path.GetFileNameWithoutExtension(model)}.onnx");
        if (File.Exists(compiledPath)) File.Delete(compiledPath);

        var sw = Stopwatch.StartNew();
        using (var compileOptions = new OrtModelCompilationOptions(so))
        {
            compileOptions.SetInputModelPath(model);
            compileOptions.SetOutputModelPath(compiledPath);
            compileOptions.CompileModel();
        }
        sw.Stop();

        compileMs = sw.Elapsed.TotalMilliseconds;
        if (File.Exists(compiledPath))
        {
            compiledBytes = new FileInfo(compiledPath).Length;
            try { File.Delete(compiledPath); } catch { /* best effort cleanup */ }
        }
        ok = true;
    }
    catch (Exception ex)
    {
        ok = false;
        error = $"0x{ex.HResult:X8} {ex.Message}".Replace("\r", " ").Replace("\n", " ");
    }

    var result = new CompileJobResult(label, model, ep, deviceType, compileMs, compiledBytes, ok, error);
    try
    {
        File.WriteAllText(outPath, JsonSerializer.Serialize(result));
    }
    catch { /* if we can't write, the parent will treat it as a crash */ }
}


// ----------------------------------------------------------------------------
// Probe mode (diagnostic)
// ----------------------------------------------------------------------------
static async Task RunProbeAsync(string[] args)
{
    Console.WriteLine("=== WinML EP Enumeration + Compile Probe ===\n");

    var envOptions = new EnvironmentCreationOptions
    {
        logId = "EpProbe",
        logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
    };
    OrtEnv.CreateInstanceWithOptions(ref envOptions);
    var ortEnv = OrtEnv.Instance();
    Console.WriteLine("OrtEnv created.");

    Console.WriteLine("\n--- Registering all installed EPs (package location + ORT RegisterExecutionProviderLibrary) ---");
    var infos = await EpRegistration.RegisterAllAsync();
    Console.WriteLine($"{"ProviderName",-34} {"Registered",-11} {"PackageFamily",-52} Notes");
    Console.WriteLine(new string('-', 130));
    foreach (var i in infos)
        Console.WriteLine($"{i.Name,-34} {i.Registered,-11} {i.PackageFamily,-52} {i.Error}");

    var devices = ortEnv.GetEpDevices();
    Console.WriteLine($"\nFound {devices.Count} EP device(s):");
    Console.WriteLine($"{"#",-3} {"EpName",-34} {"Device",-8} {"Vendor",-24}");
    Console.WriteLine(new string('-', 72));
    for (int i = 0; i < devices.Count; i++)
    {
        var d = devices[i];
        string vendor = "";
        try { vendor = d.EpVendor; } catch { }
        Console.WriteLine($"{i,-3} {d.EpName,-34} {d.HardwareDevice.Type,-8} {vendor,-24}");
    }

    var modelPath = args.Length > 0 && !args[0].StartsWith("--")
        ? Path.GetFullPath(args[0])
        : ResolveDefaultModel();
    Console.WriteLine($"\nModel: {modelPath}");
    Console.WriteLine($"Exists: {File.Exists(modelPath)}\n");
    if (!File.Exists(modelPath))
    {
        Console.WriteLine("ERROR: Model file not found. Pass a model path as the first argument.");
        return;
    }

    bool cpuOnly = !args.Contains("--all");
    bool compileOnly = args.Contains("--compile-only");
    Console.WriteLine("Timing session creation (JIT) and CompileModel (AOT) per EP"
        + (cpuOnly ? " [CPU devices only; pass --all to include GPU/NPU]" : " [ALL devices]")
        + (compileOnly ? " [compile-only: skip JIT session]" : "") + ":\n");
    Console.WriteLine($"{"EpName",-34} {"Device",-8} {"JIT session",-14} {"AOT compile",-14} Notes");
    Console.WriteLine(new string('-', 100));

    foreach (var device in devices)
    {
        if (cpuOnly && device.HardwareDevice.Type != OrtHardwareDeviceType.CPU)
        {
            Console.WriteLine($"{device.EpName,-34} {device.HardwareDevice.Type,-8} {"skipped",-14} {"skipped",-14} (non-CPU; use --all)");
            continue;
        }

        string jit = "—";
        string aot = "—";
        string notes = "";

        if (!compileOnly)
        {
            try
            {
                var so = new SessionOptions();
                so.AppendExecutionProvider(ortEnv, new[] { device }, EpOptionsFor(device.EpName));
                var sw = Stopwatch.StartNew();
                using var session = new InferenceSession(modelPath, so);
                sw.Stop();
                jit = $"{sw.ElapsedMilliseconds} ms";
            }
            catch (Exception ex)
            {
                jit = "FAIL";
                notes = $"JIT: {Short(ex)}";
            }
        }
        else
        {
            jit = "skip";
        }

        try
        {
            var so = new SessionOptions();
            so.AppendExecutionProvider(ortEnv, new[] { device }, EpOptionsFor(device.EpName));

            var epSafe = device.EpName.Replace(" ", "_").Replace(".", "_").Replace("(", "").Replace(")", "");
            var compiledPath = Path.Combine(Path.GetTempPath(), $"probe-{epSafe}-{device.HardwareDevice.Type}.onnx");
            if (File.Exists(compiledPath)) File.Delete(compiledPath);

            var sw = Stopwatch.StartNew();
            using var compileOptions = new OrtModelCompilationOptions(so);
            compileOptions.SetInputModelPath(modelPath);
            compileOptions.SetOutputModelPath(compiledPath);
            compileOptions.CompileModel();
            sw.Stop();
            aot = $"{sw.ElapsedMilliseconds} ms";
        }
        catch (Exception ex)
        {
            aot = "FAIL";
            notes = string.IsNullOrEmpty(notes) ? $"AOT: {Short(ex)}" : notes + $" | AOT: {Short(ex)}";
        }

        Console.WriteLine($"{device.EpName,-34} {device.HardwareDevice.Type,-8} {jit,-14} {aot,-14} {notes}");
    }

    Console.WriteLine("\n=== Done ===");
}


// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------
static string? GetArg(string[] args, string name)
{
    var i = Array.IndexOf(args, name);
    return (i >= 0 && i + 1 < args.Length) ? args[i + 1] : null;
}

// Per-EP provider options, mirroring the WindowsML-Lab reference (GetSessionOptions):
// each vendor EP reads its tuning knobs from session/provider options. Passing sensible
// defaults keeps parity with how the EP would be configured in a real app; unknown keys are
// ignored by EPs that don't use them.
static Dictionary<string, string> EpOptionsFor(string epName)
{
    var opts = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
    switch (epName)
    {
        case "OpenVINOExecutionProvider":
            opts["num_of_threads"] = "4";
            break;
        case "QNNExecutionProvider":
            opts["htp_performance_mode"] = "high_performance";
            break;
        // VitisAI / TensorRT-RTX / MIGraphX / CPU / DML / WinMLCG: no extra options by default.
    }
    return opts;
}

static string ResolveDefaultModel()
{
    var dir = AppContext.BaseDirectory;
    for (int i = 0; i < 8 && dir != null; i++)
    {
        var candidate = Path.Combine(dir, "Assets", "resnet50-v2-7.onnx");
        if (File.Exists(candidate)) return candidate;
        candidate = Path.Combine(dir, "WinML-EP-Benchmarking-Sample-App", "Assets", "resnet50-v2-7.onnx");
        if (File.Exists(candidate)) return candidate;
        dir = Directory.GetParent(dir)?.FullName;
    }
    return Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "Assets", "resnet50-v2-7.onnx"));
}

static string Short(Exception ex)
{
    var msg = ex.Message.Replace("\r", " ").Replace("\n", " ");
    if (msg.Length > 90) msg = msg[..90] + "…";
    return $"0x{ex.HResult:X8} {msg}";
}

record CompileJobResult(
    string label, string model, string ep, string deviceType,
    double compileMs, long compiledBytes, bool ok, string? error);

record EpDev(string ep, string deviceType, string vendor);
record EpDiag(string provider, bool registered, string packageFamily, string? error);

class EpInventoryResult
{
    public bool ok { get; set; }
    public string? error { get; set; }
    public List<EpDev> devices { get; set; } = new();
    public List<EpDiag> diagnostics { get; set; } = new();
}
