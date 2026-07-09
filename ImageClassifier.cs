using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Windows.AI.MachineLearning;
using Windows.Graphics.Imaging;
using Windows.Storage;
using System.Diagnostics;
using System.Reflection;

namespace WinMLResNet;

public class PredictionResult
{
    public string Label { get; set; } = "";
    public string Score { get; set; } = "";
}

public class ModelTensorInfo
{
    public string Name { get; set; } = "";
    public string Shape { get; set; } = "";
    public string DataType { get; set; } = "";
    public string Direction { get; set; } = "";
    public string Display => $"{Direction}: {Name}  [{Shape}]  {DataType}";
}

public class ModelInfo
{
    public string FileName { get; set; } = "";
    public string FileSize { get; set; } = "";
    public string ProducerName { get; set; } = "";
    public string GraphName { get; set; } = "";
    public string Domain { get; set; } = "";
    public string Description { get; set; } = "";
    public long ModelVersion { get; set; }
    public bool IsQuantized { get; set; }
    public string QuantizationDetail { get; set; } = "";
    public List<ModelTensorInfo> Tensors { get; set; } = new();
    public Dictionary<string, string> CustomMetadata { get; set; } = new();
}

public class RuntimeInfo
{
    public string OnnxRuntimeVersion { get; set; } = "";
    public string WindowsAppSdkVersion { get; set; } = "";
    public string WindowsMlVersion { get; set; } = "";
    public string DotNetVersion { get; set; } = "";
    public string OsVersion { get; set; } = "";
    public string Architecture { get; set; } = "";
    public string WindowsMlApi { get; set; } = "";
    public string WasdkDeployment { get; set; } = "";
    public string EpAcquisition { get; set; } = "";
}

public class EpDeviceInfo
{
    public string DisplayName { get; set; } = "";
    internal OrtEpDevice Device { get; set; } = null!;
    public override string ToString() => DisplayName;
}

public class ClassificationMetrics
{
    public int Rank { get; set; }
    public string EpName { get; set; } = "";
    public string Mode { get; set; } = "JIT";
    public double RawSessionMs { get; set; } = -1;
    public string SessionTime { get; set; } = "";
    public double RawCompileMs { get; set; } = -1;
    public string CompileTime { get; set; } = "—";
    public double RawPreprocessMs { get; set; } = -1;
    public string PreprocessTime { get; set; } = "";
    public double RawInferenceMs { get; set; } = -1;
    public string InferenceTime { get; set; } = "";
    public double RawEpPerfMs { get; set; } = -1;
    public string EpPerfTime { get; set; } = "";
    public double RawTotalMs { get; set; } = -1;
    public string TotalTime { get; set; } = "";
    public double RawMemDeltaMb { get; set; } = double.NaN;
    public string MemoryDelta { get; set; } = "";
    public string TopPrediction { get; set; } = "";
    public string TopConfidence { get; set; } = "";
}

public class ClassificationResult
{
    public List<PredictionResult> Predictions { get; set; } = new();
    public ClassificationMetrics Metrics { get; set; } = new();
}

public class ImageClassifier : IDisposable
{
    private readonly string _modelPath;
    private readonly string[] _labels;
    private bool _initialized;
    private readonly Dictionary<string, InferenceSession> _sessions = new();

    public ImageClassifier(string modelPath)
    {
        _modelPath = modelPath;
        var labelsPath = Path.Combine(AppContext.BaseDirectory, "Assets", "imagenet_labels.txt");
        _labels = File.ReadAllLines(labelsPath);

        // Initialize ORT environment early so GetModelInfo() doesn't create a conflicting one
        var envOptions = new EnvironmentCreationOptions
        {
            logId = "WinMLResNet",
            logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
        };
        OrtEnv.CreateInstanceWithOptions(ref envOptions);
    }

    public async Task InitializeAsync()
    {
        if (_initialized) return;

        // Discover which EPs are usable via a crash-isolated worker (it registers every installed EP
        // using the package-location + ORT mechanism, then enumerates devices). We then register
        // in-process ONLY the providers confirmed to produce a device, so the UI process never
        // force-loads a fragile vendor library that could native-crash on hardware without the
        // matching accelerator. The built-in CPU/DML providers need no registration.
        await EpInventory.EnsureAsync();
        var safe = EpInventory.DeviceProducingProviders
            .Where(p => !p.Equals("CPUExecutionProvider", StringComparison.OrdinalIgnoreCase)
                     && !p.Equals("DmlExecutionProvider", StringComparison.OrdinalIgnoreCase))
            .ToList();
        if (safe.Count > 0)
            await EpRegistration.RegisterSpecificAsync(safe);

        _initialized = true;
    }

    public List<EpDeviceInfo> GetAvailableDevices()
    {
        var ortEnv = OrtEnv.Instance();
        var epDevices = ortEnv.GetEpDevices();
        var result = new List<EpDeviceInfo>();

        foreach (var device in epDevices)
        {
            result.Add(new EpDeviceInfo
            {
                DisplayName = $"{device.EpName} ({device.HardwareDevice.Type})",
                Device = device
            });
        }

        return result;
    }

    public ModelInfo GetModelInfo()
    {
        var fileInfo = new FileInfo(_modelPath);
        var sizeMb = fileInfo.Length / (1024.0 * 1024.0);

        // Use a lightweight CPU session to read metadata
        using var session = new InferenceSession(_modelPath);
        var meta = session.ModelMetadata;

        var tensors = new List<ModelTensorInfo>();
        var allDataTypes = new HashSet<string>();

        foreach (var (name, nodeMeta) in session.InputMetadata)
        {
            var dtype = nodeMeta.ElementDataType.ToString();
            allDataTypes.Add(dtype);
            tensors.Add(new ModelTensorInfo
            {
                Name = name,
                Shape = string.Join(" x ", nodeMeta.Dimensions),
                DataType = dtype,
                Direction = "Input"
            });
        }

        foreach (var (name, nodeMeta) in session.OutputMetadata)
        {
            var dtype = nodeMeta.ElementDataType.ToString();
            allDataTypes.Add(dtype);
            tensors.Add(new ModelTensorInfo
            {
                Name = name,
                Shape = string.Join(" x ", nodeMeta.Dimensions),
                DataType = dtype,
                Direction = "Output"
            });
        }

        // Determine quantization from data types
        var quantizedTypes = new[] { "Int8", "UInt8", "Int4", "UInt4", "Float16" };
        var foundQuantized = allDataTypes.Where(d => quantizedTypes.Any(q => d.Contains(q, StringComparison.OrdinalIgnoreCase))).ToList();
        bool isQuantized = foundQuantized.Count > 0;
        string quantDetail = isQuantized
            ? string.Join(", ", foundQuantized)
            : "No (Float32)";

        // Check custom metadata for additional hints
        var customMeta = meta.CustomMetadataMap ?? new Dictionary<string, string>();

        return new ModelInfo
        {
            FileName = fileInfo.Name,
            FileSize = $"{sizeMb:F1} MB ({fileInfo.Length:N0} bytes)",
            ProducerName = meta.ProducerName ?? "Unknown",
            GraphName = meta.GraphName ?? "",
            Domain = meta.Domain ?? "",
            Description = meta.Description ?? "",
            ModelVersion = meta.Version,
            IsQuantized = isQuantized,
            QuantizationDetail = quantDetail,
            Tensors = tensors,
            CustomMetadata = new Dictionary<string, string>(customMeta)
        };
    }

    public static RuntimeInfo GetRuntimeInfo()
    {
        // ONNX Runtime: read native onnxruntime.dll file version (managed wrapper reports 0.0.0)
        string ortVersion = GetNativeOrtVersion();

        // Windows ML projection version: strip commit hash
        var mlInfoVer = typeof(ExecutionProviderCatalog).Assembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "Unknown";
        var mlClean = mlInfoVer.Contains('+') ? mlInfoVer[..mlInfoVer.IndexOf('+')] : mlInfoVer;

        // Detect WAS deployment mode: self-contained if core WAS DLLs exist next to the exe
        var wasDllPath = Path.Combine(AppContext.BaseDirectory, "Microsoft.WindowsAppRuntime.dll");
        bool isSelfContained = File.Exists(wasDllPath);

        // Detect WASDK version from the WindowsAppRuntime.dll file version, or fall back to the Foundation assembly
        string wasdkVersion = "Unknown";
        var runtimeDll = Path.Combine(AppContext.BaseDirectory, "Microsoft.WindowsAppRuntime.dll");
        if (File.Exists(runtimeDll))
        {
            var fvi = System.Diagnostics.FileVersionInfo.GetVersionInfo(runtimeDll);
            var pv = fvi.ProductVersion;
            if (pv != null)
            {
                // Product version is like "2.0.1" or "2.0.1+commitHash"
                wasdkVersion = pv.Contains('+') ? pv[..pv.IndexOf('+')] : pv;
            }
        }
        else
        {
            // Fallback: try the Foundation projection assembly
            try
            {
                var foundationType = Type.GetType("Microsoft.Windows.ApplicationModel.DynamicDependency.Bootstrap, Microsoft.WindowsAppSDK.Foundation");
                if (foundationType != null)
                {
                    var v = foundationType.Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "Unknown";
                    wasdkVersion = v.Contains('+') ? v[..v.IndexOf('+')] : v;
                }
            }
            catch { }
        }

        return new RuntimeInfo
        {
            OnnxRuntimeVersion = ortVersion,
            WindowsMlVersion = mlClean,
            WindowsAppSdkVersion = wasdkVersion,
            DotNetVersion = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription,
            OsVersion = $"{Environment.OSVersion.VersionString}",
            Architecture = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString(),
            WindowsMlApi = "New Windows ML (Microsoft.Windows.AI.MachineLearning)",
            WasdkDeployment = isSelfContained ? "Self-Contained" : "Framework-Dependent",
            EpAcquisition = "Evergreen (dynamic download via ExecutionProviderCatalog)"
        };
    }

    private static string GetNativeOrtVersion()
    {
        try
        {
            // Check standard location first, then NuGet runtimes/ layout (WASDK 2.0+)
            var ortDllPath = Path.Combine(AppContext.BaseDirectory, "onnxruntime.dll");
            if (!File.Exists(ortDllPath))
            {
                var rid = System.Runtime.InteropServices.RuntimeInformation.RuntimeIdentifier;
                ortDllPath = Path.Combine(AppContext.BaseDirectory, "runtimes", rid, "native", "onnxruntime.dll");
            }
            if (File.Exists(ortDllPath))
            {
                var versionInfo = System.Diagnostics.FileVersionInfo.GetVersionInfo(ortDllPath);
                // ProductVersion is like "1.23.20260219.1.4e0442d" — take major.minor
                var pv = versionInfo.ProductVersion;
                if (pv != null)
                {
                    var parts = pv.Split('.');
                    if (parts.Length >= 2)
                        return $"{parts[0]}.{parts[1]}";
                }
                return versionInfo.FileVersion ?? "Unknown";
            }
        }
        catch { }
        return "Unknown";
    }

    private (InferenceSession session, bool wasCached) GetOrCreateSession(EpDeviceInfo epDevice)
    {
        var key = $"{epDevice.DisplayName}|uncompiled";
        if (_sessions.TryGetValue(key, out var cached))
            return (cached, true);

        var ortEnv = OrtEnv.Instance();
        var sessionOptions = new SessionOptions();
        sessionOptions.AppendExecutionProvider(ortEnv, new[] { epDevice.Device }, new Dictionary<string, string>());

        var session = new InferenceSession(_modelPath, sessionOptions);
        _sessions[key] = session;
        return (session, false);
    }

    private (InferenceSession session, double compileMs, bool wasCached) GetOrCreateCompiledSession(EpDeviceInfo epDevice)
    {
        var key = $"{epDevice.DisplayName}|compiled";
        if (_sessions.TryGetValue(key, out var cached))
            return (cached, 0, true);

        var ortEnv = OrtEnv.Instance();
        var sessionOptions = new SessionOptions();
        sessionOptions.AppendExecutionProvider(ortEnv, new[] { epDevice.Device }, new Dictionary<string, string>());

        // Generate a unique compiled model path per EP
        var epSafe = epDevice.DisplayName.Replace(" ", "_").Replace("(", "").Replace(")", "");
        var compiledPath = Path.Combine(
            Path.GetDirectoryName(_modelPath)!,
            $"{Path.GetFileNameWithoutExtension(_modelPath)}-compiled-{epSafe}.onnx");

        // Always compile fresh to get accurate timing
        if (File.Exists(compiledPath))
            File.Delete(compiledPath);

        var compileSw = Stopwatch.StartNew();
        using var compileOptions = new OrtModelCompilationOptions(sessionOptions);
        compileOptions.SetInputModelPath(_modelPath);
        compileOptions.SetOutputModelPath(compiledPath);
        compileOptions.CompileModel();
        compileSw.Stop();
        double compileMs = compileSw.Elapsed.TotalMilliseconds;

        var session = new InferenceSession(compiledPath, sessionOptions);
        _sessions[key] = session;
        return (session, compileMs, false);
    }

    public async Task<ClassificationResult> ClassifyAsync(string imagePath, EpDeviceInfo epDevice, bool compiled = false)
    {
        await InitializeAsync();

        // Capture working-set baseline before any work for this run.
        var process = Process.GetCurrentProcess();
        process.Refresh();
        long memBefore = process.WorkingSet64;

        var totalSw = Stopwatch.StartNew();

        // Preprocess image
        var ppSw = Stopwatch.StartNew();
        var inputTensor = await PreprocessImageAsync(imagePath);
        ppSw.Stop();

        // Get or create session for this EP (offload to background thread — session creation and compilation are heavy)
        var sessionSw = Stopwatch.StartNew();
        InferenceSession session;
        double compileMs = 0;
        bool sessionCached = false;
        if (compiled)
        {
            (session, compileMs, sessionCached) = await Task.Run(() => GetOrCreateCompiledSession(epDevice));
        }
        else
        {
            (session, sessionCached) = await Task.Run(() => GetOrCreateSession(epDevice));
        }
        sessionSw.Stop();
        double sessionMs = sessionSw.Elapsed.TotalMilliseconds;
        bool isFirstRun = !sessionCached;

        var infSw = Stopwatch.StartNew();
        var inputName = session.InputMetadata.First().Key;
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };
        // Run inference on a background thread to avoid blocking the UI
        using var results = await Task.Run(() => session.Run(inputs));
        infSw.Stop();
        totalSw.Stop();

        // Capture working-set after inference completes (before output processing — that's negligible)
        process.Refresh();
        long memAfter = process.WorkingSet64;
        double memDeltaMb = (memAfter - memBefore) / (1024.0 * 1024.0);

        // Process output
        var outputName = session.OutputMetadata.First().Key;
        var resultTensor = results.First(r => r.Name == outputName)
            .AsEnumerable<float>().ToArray();

        // Softmax
        float maxLogit = resultTensor.Max();
        var expScores = resultTensor.Select(r => MathF.Exp(r - maxLogit)).ToArray();
        float sumExp = expScores.Sum();
        var softmax = expScores.Select(e => e / sumExp).ToArray();

        // Top-5 predictions
        var top5 = softmax
            .Select((value, index) => (value, index))
            .OrderByDescending(x => x.value)
            .Take(5)
            .ToList();

        var predictions = new List<PredictionResult>();
        foreach (var (value, index) in top5)
        {
            var label = index < _labels.Length ? _labels[index] : $"Class {index}";
            predictions.Add(new PredictionResult
            {
                Label = label,
                Score = $"{value * 100:F1}%"
            });
        }

        return new ClassificationResult
        {
            Predictions = predictions,
            Metrics = new ClassificationMetrics
            {
                EpName = epDevice.DisplayName,
                Mode = isFirstRun
                    ? (compiled ? "AOT (Cold)" : "JIT (Cold)")
                    : (compiled ? "Warm (AOT-built)" : "Warm (JIT-built)"),
                RawSessionMs = sessionMs,
                SessionTime = $"{sessionMs:F1} ms",
                RawCompileMs = compiled && !sessionCached ? compileMs : -1,
                CompileTime = compiled
                    ? (sessionCached ? "AOT cached" : $"{compileMs:F1} ms")
                    : (sessionCached ? "session cached" : "during session creation"),
                RawPreprocessMs = ppSw.Elapsed.TotalMilliseconds,
                PreprocessTime = $"{ppSw.Elapsed.TotalMilliseconds:F1} ms",
                RawInferenceMs = infSw.Elapsed.TotalMilliseconds,
                InferenceTime = $"{infSw.Elapsed.TotalMilliseconds:F1} ms",
                RawEpPerfMs = sessionMs + (compiled && !sessionCached ? compileMs : 0) + infSw.Elapsed.TotalMilliseconds,
                EpPerfTime = $"{(sessionMs + (compiled && !sessionCached ? compileMs : 0) + infSw.Elapsed.TotalMilliseconds):F1} ms",
                RawTotalMs = totalSw.Elapsed.TotalMilliseconds,
                TotalTime = $"{totalSw.Elapsed.TotalMilliseconds:F1} ms",
                RawMemDeltaMb = memDeltaMb,
                MemoryDelta = FormatMemDelta(memDeltaMb, isFirstRun),
                TopPrediction = predictions.First().Label,
                TopConfidence = predictions.First().Score
            }
        };
    }

    internal static string FormatMemDelta(double mb, bool isFirstRun = false)
    {
        if (double.IsNaN(mb)) return "";
        // Show negative deltas (e.g., GC freed memory) too
        var prefix = mb >= 0 ? "+" : "";
        var marker = isFirstRun ? " *" : "";
        return $"{prefix}{mb:F1} MB{marker}";
    }

    private static async Task<DenseTensor<float>> PreprocessImageAsync(string imagePath)
    {
        var file = await StorageFile.GetFileFromPathAsync(imagePath);
        using var stream = await file.OpenReadAsync();
        var decoder = await BitmapDecoder.CreateAsync(stream);

        var transform = new BitmapTransform
        {
            ScaledWidth = 224,
            ScaledHeight = 224,
            InterpolationMode = BitmapInterpolationMode.Linear
        };
        var pixelData = await decoder.GetPixelDataAsync(
            BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied,
            transform, ExifOrientationMode.IgnoreExifOrientation,
            ColorManagementMode.DoNotColorManage);
        var pixels = pixelData.DetachPixelData();

        float[] mean = [0.485f, 0.456f, 0.406f];
        float[] std = [0.229f, 0.224f, 0.225f];

        var tensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
        for (int y = 0; y < 224; y++)
        {
            for (int x = 0; x < 224; x++)
            {
                int pixelIndex = (y * 224 + x) * 4; // BGRA
                float b = pixels[pixelIndex] / 255f;
                float g = pixels[pixelIndex + 1] / 255f;
                float r = pixels[pixelIndex + 2] / 255f;

                tensor[0, 0, y, x] = (r - mean[0]) / std[0];
                tensor[0, 1, y, x] = (g - mean[1]) / std[1];
                tensor[0, 2, y, x] = (b - mean[2]) / std[2];
            }
        }

        return tensor;
    }

    public void Dispose()
    {
        foreach (var session in _sessions.Values)
            session.Dispose();
        _sessions.Clear();
    }

    /// <summary>
    /// Disposes all cached InferenceSessions so the next ClassifyAsync call for any EP+Mode
    /// will be a true 'cold' run (model loaded into memory fresh).
    /// </summary>
    public void ClearSessionCache()
    {
        foreach (var session in _sessions.Values)
            session.Dispose();
        _sessions.Clear();
        // Encourage the GC to reclaim native handles before the next run begins.
        GC.Collect();
        GC.WaitForPendingFinalizers();
    }
}
