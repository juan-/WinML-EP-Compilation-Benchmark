using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Windows.Storage.Pickers;
using Microsoft.UI.Xaml.Media.Imaging;
using System.Collections.ObjectModel;
using Microsoft.ML.OnnxRuntime;

namespace WinMLResNet;

public sealed partial class MainWindow : Window
{
    private ImageClassifier _classifier;
    private readonly ObservableCollection<PredictionResult> _results = new();
    private readonly ObservableCollection<ClassificationMetrics> _metrics = new();
    private string? _currentImagePath;
    private List<EpDeviceInfo> _devices = new();
    private string _sortColumn = "EpLatency";
    private bool _sortAscending = true;

    // Embedding tab state
    private TextEmbedder? _embedder;
    private readonly ObservableCollection<EmbeddingMatch> _embMatches = new();
    private readonly ObservableCollection<EmbeddingMetrics> _embMetrics = new();
    private List<EpDeviceInfo> _embDevices = new();
    private string _embSortColumn = "EpLatency";
    private bool _embSortAscending = true;
    private bool _embAutoDownloadAttempted;

    // EP Compilation Benchmark tab state
    private readonly ObservableCollection<CompileModelItem> _compileModels = new();
    private readonly ObservableCollection<CompileEpItem> _compileEps = new();
    private readonly ObservableCollection<CompileResultRow> _compileResults = new();
    private readonly ObservableCollection<EpDiagnosticRow> _epDiagnostics = new();
    private bool _compileEpsPopulated;

    // Column header labels (without arrows) for reset
    private static readonly Dictionary<string, string> ColumnLabels = new()
    {
        ["EpName"] = "Execution Provider",
        ["Mode"] = "Mode",
        ["Preprocess"] = "Image Preprocessing",
        ["Session"] = "Session Creation",
        ["Compile"] = "AOT Compile",
        ["Inference"] = "Inference",
        ["EpLatency"] = "EP Total",
        ["Total"] = "Total",
        ["Memory"] = "Memory Δ",
        ["TopPrediction"] = "Top Prediction",
        ["Confidence"] = "Confidence"
    };

    private static readonly Dictionary<string, string> EmbColumnLabels = new()
    {
        ["EpName"] = "Execution Provider",
        ["Mode"] = "Mode",
        ["Tokenize"] = "Tokenization",
        ["Session"] = "Session Creation",
        ["Compile"] = "AOT Compile",
        ["Inference"] = "Inference",
        ["EpLatency"] = "EP Total",
        ["Total"] = "Total",
        ["Memory"] = "Memory Δ",
        ["TopMatch"] = "Top Match",
        ["Similarity"] = "Similarity"
    };

    public MainWindow()
    {
        this.InitializeComponent();

        // Set window icon
        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(this);
        var windowId = Microsoft.UI.Win32Interop.GetWindowIdFromWindow(hwnd);
        var appWindow = Microsoft.UI.Windowing.AppWindow.GetFromWindowId(windowId);
        appWindow.SetIcon(Path.Combine(AppContext.BaseDirectory, "Assets", "app-icon.ico"));

        ResultsListView.ItemsSource = _results;
        MetricsListView.ItemsSource = _metrics;
        EmbMatchesListView.ItemsSource = _embMatches;
        EmbMetricsListView.ItemsSource = _embMetrics;

        var modelPath = Path.Combine(AppContext.BaseDirectory, "Assets", "resnet50-v2-7.onnx");
        _classifier = new ImageClassifier(modelPath);

        UpdateModelNameDisplay(modelPath);
        PopulateModelInfo();
        PopulateRuntimeInfo();
        InitializeEmbeddingsTabUi();
        InitializeCompileTabUi();

        // WASDK 2.0 Pivot doesn't constrain content height properly — fix ScrollViewer on resize
        this.SizeChanged += (s, e) => ApplyLayoutConstraints(e.Size.Height);

        // Also apply on first load
        this.Activated += (s, e) =>
        {
            if (e.WindowActivationState != WindowActivationState.Deactivated)
                ApplyLayoutConstraints(this.Bounds.Height);
        };

        StatusText.Text = "Initializing Windows ML...";
        _ = InitializeClassifierAsync();
        _ = TryInitializeEmbedderAsync();
    }

    private void PopulateModelInfo()
    {
        try
        {
            var info = _classifier.GetModelInfo();
            ModelFileName.Text = info.FileName;
            ModelFileSize.Text = info.FileSize;
            ModelProducer.Text = info.ProducerName;
            ModelVersion.Text = info.ModelVersion.ToString();
            ModelQuantized.Text = info.IsQuantized ? $"Yes ({info.QuantizationDetail})" : info.QuantizationDetail;
            ModelDomain.Text = string.IsNullOrEmpty(info.Domain) ? "—" : info.Domain;
            ModelDescription.Text = string.IsNullOrEmpty(info.Description) ? "—" : info.Description;
            TensorListView.ItemsSource = info.Tensors;
        }
        catch (Exception ex)
        {
            ModelFileName.Text = $"Error: {ex.Message}";
        }
    }

    private void PopulateEmbModelInfo()
    {
        if (_embedder is null) return;
        try
        {
            var info = _embedder.GetModelInfo();
            EmbModelFileName.Text = info.FileName;
            EmbModelFileSize.Text = info.FileSize;
            EmbModelProducer.Text = info.ProducerName;
            EmbModelVersion.Text = info.ModelVersion.ToString();
            EmbModelQuantized.Text = info.IsQuantized ? $"Yes ({info.QuantizationDetail})" : info.QuantizationDetail;
            EmbModelDomain.Text = string.IsNullOrEmpty(info.Domain) ? "—" : info.Domain;
            EmbModelDescription.Text = string.IsNullOrEmpty(info.Description) ? "—" : info.Description;
            EmbTensorListView.ItemsSource = info.Tensors;

            EmbModelInfoStatus.Visibility = Microsoft.UI.Xaml.Visibility.Collapsed;
            EmbModelInfoGrid.Visibility = Microsoft.UI.Xaml.Visibility.Visible;
            EmbModelTensorHeader.Visibility = Microsoft.UI.Xaml.Visibility.Visible;
            EmbTensorListView.Visibility = Microsoft.UI.Xaml.Visibility.Visible;
        }
        catch (Exception ex)
        {
            EmbModelInfoStatus.Text = $"Error: {ex.Message}";
        }
    }

    private void PopulateRuntimeInfo()
    {
        var rt = ImageClassifier.GetRuntimeInfo();
        VersionWAS.Text = rt.WindowsAppSdkVersion;
        VersionORT.Text = rt.OnnxRuntimeVersion;
        VersionWinML.Text = rt.WindowsMlVersion;
        VersionDotNet.Text = rt.DotNetVersion;
        VersionOS.Text = rt.OsVersion;
        VersionArch.Text = rt.Architecture;
        DeployMlApi.Text = rt.WindowsMlApi;
        DeployWasMode.Text = rt.WasdkDeployment;
        DeployEpMode.Text = rt.EpAcquisition;
    }

    private async Task InitializeClassifierAsync()
    {
        try
        {
            await _classifier.InitializeAsync();
            _devices = _classifier.GetAvailableDevices();
            EpSelector.ItemsSource = _devices;
            if (_devices.Count > 0)
                EpSelector.SelectedIndex = 0;
            StatusText.Text = $"Ready — {_devices.Count} EP device(s) available.";

            // EPs are global to the OrtEnv — share the list with the Embeddings tab so the dropdown
            // is populated immediately, even before the embedding model is downloaded.
            PopulateEmbDevicesFromShared();

            // Populate the compilation-benchmark EP checklist now that all EPs are registered.
            PopulateCompileEps();
        }
        catch (Exception ex)
        {
            StatusText.Text = $"Init error: {ex.Message}";
        }
    }

    private async void OnPickImage(object sender, RoutedEventArgs e)
    {
        var picker = new FileOpenPicker();
        picker.FileTypeFilter.Add(".jpg");
        picker.FileTypeFilter.Add(".jpeg");
        picker.FileTypeFilter.Add(".png");
        picker.FileTypeFilter.Add(".bmp");

        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(this);
        WinRT.Interop.InitializeWithWindow.Initialize(picker, hwnd);

        var file = await picker.PickSingleFileAsync();
        if (file is null) return;

        _currentImagePath = file.Path;

        // Show preview
        var bitmapImage = new BitmapImage();
        using (var stream = await file.OpenReadAsync())
        {
            await bitmapImage.SetSourceAsync(stream);
        }
        PreviewImage.Source = bitmapImage;

        ClassifyButton.IsEnabled = true;
        RunAllButton.IsEnabled = true;
        StatusText.Text = $"Image loaded: {file.Name}";
    }

    private async void OnClassify(object sender, RoutedEventArgs e)
    {
        if (_currentImagePath is null || EpSelector.SelectedItem is not EpDeviceInfo device) return;

        SetButtonsEnabled(false);
        StatusText.Text = $"Classifying with {device.DisplayName}...";

        try
        {
            // Remove any existing rows for this EP so we get a clean cold/warm pair (cold requires a fresh session).
            // Note: removing from _metrics doesn't evict the cached InferenceSession, so we can't truly force a "cold"
            // run again without restarting the app. The first time the user picks an EP, that run is cold; subsequent
            // single-EP clicks of the same EP will all show as warm. Compare All EPs (which clears caches at start)
            // gives the proper cold/warm pair.
            var existing = _metrics.Where(m => m.EpName == device.DisplayName).ToList();
            foreach (var item in existing) _metrics.Remove(item);

            // Run JIT twice (cold + warm). If the session is already cached, both rows will be Warm.
            for (int pass = 0; pass < 2; pass++)
            {
                StatusText.Text = $"Classifying with {device.DisplayName} (JIT, {(pass == 0 ? "cold" : "warm")})...";
                var result = await _classifier.ClassifyAsync(_currentImagePath, device, compiled: false);
                _metrics.Add(result.Metrics);
                if (pass == 0)
                {
                    _results.Clear();
                    foreach (var p in result.Predictions) _results.Add(p);
                }
            }

            // Run AOT twice (cold + warm)
            for (int pass = 0; pass < 2; pass++)
            {
                StatusText.Text = $"Compiling (AOT) & classifying with {device.DisplayName} ({(pass == 0 ? "cold" : "warm")})...";
                try
                {
                    var compiledResult = await _classifier.ClassifyAsync(_currentImagePath, device, compiled: true);
                    _metrics.Add(compiledResult.Metrics);
                }
                catch (Exception compEx)
                {
                    _metrics.Add(new ClassificationMetrics
                    {
                        EpName = device.DisplayName,
                        Mode = pass == 0 ? "AOT (Cold)" : "Warm (AOT-built)",
                        CompileTime = "Error",
                        InferenceTime = "—",
                        TotalTime = ExtractErrorCode(compEx)
                    });
                    break; // if cold compile failed, warm will too
                }
            }

            ApplyInferenceHeatMap();
            StatusText.Text = $"Done — {device.DisplayName} benchmarked (4 runs).";
        }
        catch (Exception ex)
        {
            StatusText.Text = $"Error: {ex.Message}";
        }
        finally
        {
            SetButtonsEnabled(true);
        }
    }

    private async void OnRunAll(object sender, RoutedEventArgs e)
    {
        if (_currentImagePath is null) return;

        SetButtonsEnabled(false);
        _metrics.Clear();
        // Clear session cache so every EP starts with a true cold run.
        _classifier.ClearSessionCache();
        // 4 runs per EP: JIT cold, JIT warm, AOT cold, AOT warm
        int total = _devices.Count * 4;
        int step = 0;

        for (int i = 0; i < _devices.Count; i++)
        {
            var device = _devices[i];

            // JIT — cold + warm (back-to-back so 'Warm' actually hits the cache)
            for (int pass = 0; pass < 2; pass++)
            {
                step++;
                var passLabel = pass == 0 ? "cold" : "warm";
                StatusText.Text = $"[{step}/{total}] {device.DisplayName} (JIT, {passLabel})...";
                try
                {
                    var result = await _classifier.ClassifyAsync(_currentImagePath, device, compiled: false);
                    _metrics.Add(result.Metrics);
                    ApplyInferenceHeatMap();

                    // Update predictions after every EP so they appear immediately
                    _results.Clear();
                    foreach (var p in result.Predictions)
                        _results.Add(p);
                }
                catch (Exception ex)
                {
                    _metrics.Add(new ClassificationMetrics
                    {
                        EpName = device.DisplayName,
                        Mode = pass == 0 ? "JIT (Cold)" : "Warm (JIT-built)",
                        InferenceTime = "Error",
                        TotalTime = ExtractErrorCode(ex)
                    });
                    ApplyInferenceHeatMap();
                }
            }

            // AOT — cold + warm
            for (int pass = 0; pass < 2; pass++)
            {
                step++;
                var passLabel = pass == 0 ? "cold" : "warm";
                StatusText.Text = $"[{step}/{total}] {device.DisplayName} (AOT, {passLabel})...";
                try
                {
                    var compiledResult = await _classifier.ClassifyAsync(_currentImagePath, device, compiled: true);
                    _metrics.Add(compiledResult.Metrics);
                    ApplyInferenceHeatMap();
                }
                catch (Exception ex)
                {
                    _metrics.Add(new ClassificationMetrics
                    {
                        EpName = device.DisplayName,
                        Mode = pass == 0 ? "AOT (Cold)" : "Warm (AOT-built)",
                        CompileTime = "Error",
                        InferenceTime = "—",
                        TotalTime = ExtractErrorCode(ex)
                    });
                    ApplyInferenceHeatMap();
                }
            }
        }

        StatusText.Text = $"Comparison complete — {_devices.Count} EP(s) x 4 runs benchmarked.";
        SetButtonsEnabled(true);
    }

    private void SetButtonsEnabled(bool enabled)
    {
        PickImageButton.IsEnabled = enabled;
        ClassifyButton.IsEnabled = enabled && _currentImagePath != null;
        RunAllButton.IsEnabled = enabled && _currentImagePath != null;
        ExportButton.IsEnabled = _metrics.Count > 0;
    }

    private async void OnExportCsv(object sender, RoutedEventArgs e)
    {
        if (_metrics.Count == 0) return;

        var picker = new Windows.Storage.Pickers.FileSavePicker();
        picker.SuggestedStartLocation = Windows.Storage.Pickers.PickerLocationId.DocumentsLibrary;
        picker.FileTypeChoices.Add("CSV", new List<string> { ".csv" });
        picker.SuggestedFileName = $"EP_Benchmark_{DateTime.Now:yyyyMMdd_HHmmss}";

        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(this);
        WinRT.Interop.InitializeWithWindow.Initialize(picker, hwnd);

        var file = await picker.PickSaveFileAsync();
        if (file is null) return;

        var lines = new List<string>
        {
            "Rank,Execution Provider,Mode,Image Preprocessing (ms),Session Creation (ms),AOT Compile (ms),Inference (ms),EP Total (ms),Total (ms),Memory Delta (MB),Top Prediction,Confidence"
        };

        foreach (var m in _metrics)
        {
            var preprocess = m.RawPreprocessMs >= 0 ? $"{m.RawPreprocessMs:F1}" : "";
            var session = m.RawSessionMs >= 0 ? $"{m.RawSessionMs:F1}" : "";
            var compile = m.RawCompileMs >= 0 ? $"{m.RawCompileMs:F1}" : m.CompileTime.Replace("—", "").Trim();
            var inference = m.RawInferenceMs >= 0 ? $"{m.RawInferenceMs:F1}" : "Error";
            var epLatency = m.RawEpPerfMs >= 0 ? $"{m.RawEpPerfMs:F1}" : "Error";
            var total = m.RawTotalMs >= 0 ? $"{m.RawTotalMs:F1}" : "Error";
            var memory = double.IsNaN(m.RawMemDeltaMb) ? "" : $"{m.RawMemDeltaMb:F1}";
            lines.Add($"{m.Rank},\"{m.EpName}\",{m.Mode},{preprocess},{session},{compile},{inference},{epLatency},{total},{memory},\"{m.TopPrediction}\",{m.TopConfidence}");
        }

        await Windows.Storage.FileIO.WriteLinesAsync(file, lines);
        StatusText.Text = $"Exported {_metrics.Count} rows to {file.Name}";
    }

    private void OnColumnHeaderClick(object sender, RoutedEventArgs e)
    {
        if (sender is not Button btn || btn.Tag is not string col) return;
        if (_metrics.Count == 0) return;

        if (_sortColumn == col)
            _sortAscending = !_sortAscending;
        else
        {
            _sortColumn = col;
            _sortAscending = true;
        }

        UpdateColumnHeaderArrows();
        ApplyInferenceHeatMap();
    }

    private void UpdateColumnHeaderArrows()
    {
        var arrow = _sortAscending ? " ▲" : " ▼";
        var headers = new Dictionary<string, TextBlock>
        {
            ["EpName"] = Header_EpName,
            ["Mode"] = Header_Mode,
            ["Preprocess"] = Header_Preprocess,
            ["Session"] = Header_Session,
            ["Compile"] = Header_Compile,
            ["Inference"] = Header_Inference,
            ["EpLatency"] = Header_EpLatency,
            ["Total"] = Header_Total,
            ["Memory"] = Header_Memory,
            ["TopPrediction"] = Header_TopPrediction,
            ["Confidence"] = Header_Confidence
        };
        foreach (var (key, tb) in headers)
            tb.Text = key == _sortColumn ? ColumnLabels[key] + arrow : ColumnLabels[key];
    }

    private async void OnDownloadModel(object sender, RoutedEventArgs e)
    {
        await Windows.System.Launcher.LaunchUriAsync(new Uri("https://github.com/onnx/models"));
    }

    private async void OnChangeModel(object sender, RoutedEventArgs e)
    {
        var picker = new FileOpenPicker();
        picker.FileTypeFilter.Add(".onnx");

        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(this);
        WinRT.Interop.InitializeWithWindow.Initialize(picker, hwnd);

        var file = await picker.PickSingleFileAsync();
        if (file is null) return;

        // Dispose old classifier and create new one
        _classifier.Dispose();
        _classifier = new ImageClassifier(file.Path);

        // Update UI
        UpdateModelNameDisplay(file.Path);
        _results.Clear();
        _metrics.Clear();
        PopulateModelInfo();

        // Re-initialize
        StatusText.Text = $"Model changed to {file.Name}. Initializing...";
        await InitializeClassifierAsync();
    }

    private void UpdateModelNameDisplay(string modelPath)
    {
        var fi = new FileInfo(modelPath);
        var sizeMb = fi.Length / (1024.0 * 1024.0);
        var sizeText = sizeMb >= 1024 ? $"{sizeMb / 1024.0:F1} GB" : $"{sizeMb:F1} MB";
        ModelNameText.Text = $"{fi.Name} ({sizeText})";
    }

    // ===== EP Compilation Benchmark tab =====

    private void InitializeCompileTabUi()
    {
        CompileModelsList.ItemsSource = _compileModels;
        CompileEpList.ItemsSource = _compileEps;
        CompileResultsListView.ItemsSource = _compileResults;
        CompileEpDiagnosticsList.ItemsSource = _epDiagnostics;

        // Seed with the bundled models that are present on disk.
        AddCompileModel(Path.Combine(AppContext.BaseDirectory, "Assets", "resnet50-v2-7.onnx"));
        AddCompileModel(Path.Combine(AppContext.BaseDirectory, "Assets", TextEmbedder.DefaultModelFileName));

        // Auto-discover any models under Assets/certification-models (cert samples + exported EG
        // models). These large external-data models are intentionally NOT copied into bin, so look
        // in the output Assets folder first and then walk up to the source project Assets folder.
        foreach (var certDir in EnumerateCertModelDirs())
        {
            if (!Directory.Exists(certDir)) continue;
            foreach (var onnx in Directory.EnumerateFiles(certDir, "*.onnx", SearchOption.AllDirectories))
                AddCompileModel(onnx);
        }

        UpdateCompileModelsPlaceholder();
    }

    private static IEnumerable<string> EnumerateCertModelDirs()
    {
        // Output copy (if any), then the source project Assets folder resolved by walking up.
        yield return Path.Combine(AppContext.BaseDirectory, "Assets", "certification-models");
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        for (int i = 0; i < 6 && dir is not null; i++)
        {
            var candidate = Path.Combine(dir.FullName, "Assets", "certification-models");
            if (Directory.Exists(candidate)) { yield return candidate; break; }
            dir = dir.Parent;
        }
    }

    private void AddCompileModel(string path)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path)) return;
        var full = Path.GetFullPath(path);
        if (_compileModels.Any(m => string.Equals(m.Path, full, StringComparison.OrdinalIgnoreCase))) return;
        var fi = new FileInfo(full);

        // Exports and ASG drops often name every file "model.onnx" inside a per-model subfolder,
        // so qualify a generic filename with its parent folder to keep rows distinguishable.
        var label = fi.Name;
        if (fi.Name.Equals("model.onnx", StringComparison.OrdinalIgnoreCase) && fi.Directory is not null)
            label = $"{fi.Directory.Name}/{fi.Name}";

        // Include external-data weights (model.onnx.data) in the reported size so it's not misleading.
        long size = fi.Length;
        foreach (var data in Directory.EnumerateFiles(fi.DirectoryName!, $"{fi.Name}.data", SearchOption.TopDirectoryOnly))
            size += new FileInfo(data).Length;

        _compileModels.Add(new CompileModelItem { Label = label, Path = full, SizeBytes = size, Include = true });
    }

    private void UpdateCompileModelsPlaceholder()
    {
        CompileNoModelsText.Visibility = _compileModels.Count == 0 ? Visibility.Visible : Visibility.Collapsed;
    }

    private async void PopulateCompileEps()
    {
        try
        {
            // Source the EP list from the crash-isolated inventory worker rather than registering
            // vendor EPs in this (UI) process — force-loading a fragile vendor library here can
            // native-crash the app.
            await EpInventory.EnsureAsync();

            _compileEps.Clear();
            foreach (var d in EpInventory.Devices)
            {
                var type = d.DeviceType;
                bool isCpu = string.Equals(type, "CPU", StringComparison.OrdinalIgnoreCase);
                _compileEps.Add(new CompileEpItem
                {
                    EpName = d.Ep,
                    DeviceType = type,
                    DisplayName = $"{d.Ep} ({type})",
                    Include = isCpu,
                    Note = isCpu ? "" : "runs isolated — may crash without matching hardware"
                });
            }

            // Also offer registered vendor providers that didn't enumerate a device on this
            // hardware (e.g. OpenVINO CPU on a non-Intel box). Selecting one attempts the real
            // vendor path — OpenVINO's own CPU runtime, not the built-in MLAS CPU EP — and reports
            // honestly if the device isn't available here. On matching hardware these just work.
            foreach (var t in EpInventory.RegisteredButUnenumeratedTargets())
            {
                _compileEps.Add(new CompileEpItem
                {
                    EpName = t.Ep,
                    DeviceType = t.DeviceType,
                    DisplayName = $"{t.Ep} ({t.DeviceType})",
                    Include = false,
                    Note = "vendor target — needs matching hardware to enumerate a device"
                });
            }

            _compileEpsPopulated = true;

            PopulateEpDiagnostics();
        }
        catch (Exception ex)
        {
            CompileStatusText.Text = $"Could not enumerate EPs: {ex.Message}";
        }
    }

    /// <summary>
    /// Builds the read-only "installed provider diagnostics" list, explaining for every provider
    /// whether it registered and whether it produced a selectable device. This is what makes the
    /// "why do I only see N EPs?" question self-service: registered providers that enumerate no
    /// compatible hardware, and providers whose native library failed to load, are shown with the
    /// exact reason.
    /// </summary>
    private void PopulateEpDiagnostics()
    {
        _epDiagnostics.Clear();

        // Provider name -> device types it enumerated (these are the selectable ones).
        var deviceTypesByEp = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase);
        foreach (var d in EpInventory.Devices)
        {
            if (!deviceTypesByEp.TryGetValue(d.Ep, out var list))
                deviceTypesByEp[d.Ep] = list = new List<string>();
            if (!list.Contains(d.DeviceType)) list.Add(d.DeviceType);
        }

        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // 1. Selectable providers (have a device) — includes built-ins CPU/DML.
        foreach (var (ep, types) in deviceTypesByEp.OrderBy(k => k.Key, StringComparer.OrdinalIgnoreCase))
        {
            seen.Add(ep);
            _epDiagnostics.Add(new EpDiagnosticRow
            {
                Provider = ep,
                Status = "✓ Selectable",
                Detail = $"device: {string.Join(", ", types)}"
            });
        }

        // 2. Providers whose library registered but enumerated no compatible device, and providers
        //    whose library failed to load — from the registration pass.
        foreach (var info in EpInventory.Diagnostics)
        {
            if (seen.Contains(info.Provider)) continue;
            seen.Add(info.Provider);

            if (info.Registered)
            {
                _epDiagnostics.Add(new EpDiagnosticRow
                {
                    Provider = info.Provider,
                    Status = "○ Registered",
                    Detail = "library loaded, but no compatible hardware device is present"
                });
            }
            else
            {
                _epDiagnostics.Add(new EpDiagnosticRow
                {
                    Provider = info.Provider,
                    Status = "✗ Not loaded",
                    Detail = string.IsNullOrEmpty(info.Error) ? "provider library did not load" : info.Error!
                });
            }
        }
    }

    private async void OnCompileAddFile(object sender, RoutedEventArgs e)
    {
        var picker = new FileOpenPicker();
        picker.FileTypeFilter.Add(".onnx");
        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(this);
        WinRT.Interop.InitializeWithWindow.Initialize(picker, hwnd);
        var files = await picker.PickMultipleFilesAsync();
        if (files is null || files.Count == 0) return;
        foreach (var f in files) AddCompileModel(f.Path);
        UpdateCompileModelsPlaceholder();
    }

    private async void OnCompileAddFolder(object sender, RoutedEventArgs e)
    {
        var picker = new FolderPicker();
        picker.FileTypeFilter.Add("*");
        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(this);
        WinRT.Interop.InitializeWithWindow.Initialize(picker, hwnd);
        var folder = await picker.PickSingleFolderAsync();
        if (folder is null) return;
        int added = 0;
        foreach (var file in Directory.EnumerateFiles(folder.Path, "*.onnx", SearchOption.AllDirectories))
        {
            int before = _compileModels.Count;
            AddCompileModel(file);
            if (_compileModels.Count > before) added++;
        }
        UpdateCompileModelsPlaceholder();
        CompileStatusText.Text = $"Added {added} model(s) from {folder.Name}.";
    }

    private void OnCompileClearModels(object sender, RoutedEventArgs e)
    {
        _compileModels.Clear();
        UpdateCompileModelsPlaceholder();
    }

    private void OnCompileRefreshEps(object sender, RoutedEventArgs e)
    {
        PopulateCompileEps();
        CompileStatusText.Text = $"{_compileEps.Count} EP device(s) available.";
    }

    private async void OnRunCompileBenchmark(object sender, RoutedEventArgs e)
    {
        if (!_compileEpsPopulated) PopulateCompileEps();

        var models = _compileModels.Where(m => m.Include).ToList();
        var eps = _compileEps.Where(ep => ep.Include).ToList();
        if (models.Count == 0) { CompileStatusText.Text = "Add and check at least one model."; return; }
        if (eps.Count == 0) { CompileStatusText.Text = "Check at least one execution provider."; return; }

        var worker = CompileBenchmarkRunner.LocateWorker();
        if (worker is null)
        {
            CompileStatusText.Text = "Compile worker not found — build the CompileErrorTest project (same platform/config as this app).";
            return;
        }

        SetCompileButtonsEnabled(false);
        _compileResults.Clear();

        int step = 0;
        int total = models.Count * eps.Count;
        foreach (var model in models)
        {
            foreach (var ep in eps)
            {
                step++;
                var row = new CompileResultRow
                {
                    Rank = step,
                    ModelLabel = model.Label,
                    EpName = ep.EpName,
                    DeviceType = ep.DeviceType,
                    CompileText = "…",
                    Status = "Running"
                };
                _compileResults.Add(row);
                CompileStatusText.Text = $"[{step}/{total}] {model.Label} × {ep.DisplayName}…";

                var res = await CompileBenchmarkRunner.RunAsync(worker, model.Path, ep.EpName, ep.DeviceType, model.Label);

                if (res.Ok)
                {
                    row.RawCompileMs = res.CompileMs;
                    row.CompiledBytes = res.CompiledBytes;
                    row.CompileText = $"{res.CompileMs:F1} ms";
                    row.SizeText = res.CompiledBytes > 0 ? $"{res.CompiledBytes / (1024.0 * 1024.0):F2} MB" : "—";
                    row.Status = "OK";
                }
                else if (res.Crashed)
                {
                    row.CompileText = "—";
                    row.Status = "Crashed";
                    row.Error = res.Error;
                }
                else
                {
                    row.CompileText = "—";
                    row.Status = "Error";
                    row.Error = res.Error;
                }
            }
        }

        CompileStatusText.Text = $"Done — {total} compile job(s): {models.Count} model(s) × {eps.Count} EP(s).";
        CompileExportButton.IsEnabled = _compileResults.Count > 0;
        SetCompileButtonsEnabled(true);
    }

    private void SetCompileButtonsEnabled(bool enabled)
    {
        CompileRunButton.IsEnabled = enabled;
        CompileAddFileButton.IsEnabled = enabled;
        CompileAddFolderButton.IsEnabled = enabled;
        CompileClearModelsButton.IsEnabled = enabled;
        CompileRefreshEpsButton.IsEnabled = enabled;
    }

    private async void OnCompileExportCsv(object sender, RoutedEventArgs e)
    {
        if (_compileResults.Count == 0) return;
        var picker = new FileSavePicker();
        picker.SuggestedStartLocation = PickerLocationId.DocumentsLibrary;
        picker.FileTypeChoices.Add("CSV", new List<string> { ".csv" });
        picker.SuggestedFileName = $"EP_Compile_Benchmark_{DateTime.Now:yyyyMMdd_HHmmss}";
        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(this);
        WinRT.Interop.InitializeWithWindow.Initialize(picker, hwnd);
        var file = await picker.PickSaveFileAsync();
        if (file is null) return;

        var lines = new List<string> { "#,Model,Execution Provider,Device,Compile Time (ms),Compiled Size (bytes),Status,Detail" };
        foreach (var r in _compileResults)
        {
            var ms = r.RawCompileMs >= 0 ? r.RawCompileMs.ToString("F1") : "";
            var bytes = r.CompiledBytes >= 0 ? r.CompiledBytes.ToString() : "";
            var detail = (r.Error ?? "").Replace("\"", "'");
            lines.Add($"{r.Rank},\"{r.ModelLabel}\",\"{r.EpName}\",{r.DeviceType},{ms},{bytes},{r.Status},\"{detail}\"");
        }
        await Windows.Storage.FileIO.WriteLinesAsync(file, lines);
        CompileStatusText.Text = $"Exported {_compileResults.Count} row(s) to {file.Name}";
    }

    private void ApplyLayoutConstraints(double windowHeight)
    {
        var availableHeight = windowHeight - 160;
        if (availableHeight > 100)
            ClassificationScrollViewer.MaxHeight = availableHeight;
        ImagePreviewGrid.MaxHeight = windowHeight * 0.30;
    }

    private static string ExtractErrorCode(Exception ex)
    {
        var msg = ex.Message;
        var match = System.Text.RegularExpressions.Regex.Match(msg, @"\[ErrorCode:(\w+)\]");
        if (match.Success)
            return $"\u274C Error ({match.Groups[1].Value})";
        var short_ = msg.Length > 30 ? msg[..30] + "…" : msg;
        return $"\u274C Error ({short_})";
    }

    private void ApplyInferenceHeatMap()
    {
        // Apply heat map emoji to inference and Session+Compile+Inference columns
        // (skip AOT Compile — only one row per EP carries a measured value, so heat-map there is misleading)
        ApplyHeatMapToColumn(
            m => m.RawInferenceMs,
            (m, text) => m.InferenceTime = text,
            m => m.RawInferenceMs >= 0);
        ApplyHeatMapToColumn(
            m => m.RawEpPerfMs,
            (m, text) => m.EpPerfTime = text,
            m => m.RawEpPerfMs >= 0);

        // Sort by selected column; errors always go to bottom
        Func<ClassificationMetrics, object> sortKey = _sortColumn switch
        {
            "EpName" => m => m.EpName,
            "Mode" => m => m.Mode,
            "Preprocess" => m => m.RawPreprocessMs < 0 ? (object)double.MaxValue : m.RawPreprocessMs,
            "Session" => m => m.RawSessionMs < 0 ? (object)double.MaxValue : m.RawSessionMs,
            "Compile" => m => m.RawCompileMs < 0 ? (object)double.MaxValue : m.RawCompileMs,
            "Inference" => m => m.RawInferenceMs < 0 ? (object)double.MaxValue : m.RawInferenceMs,
            "EpLatency" => m => m.RawEpPerfMs < 0 ? (object)double.MaxValue : m.RawEpPerfMs,
            "Total" => m => m.RawTotalMs < 0 ? (object)double.MaxValue : m.RawTotalMs,
            "Memory" => m => double.IsNaN(m.RawMemDeltaMb) ? (object)double.MaxValue : m.RawMemDeltaMb,
            "TopPrediction" => m => m.TopPrediction,
            "Confidence" => m => m.TopConfidence,
            _ => m => m.RawEpPerfMs < 0 ? (object)double.MaxValue : m.RawEpPerfMs
        };

        int rank = 0;
        var sorted = _sortAscending
            ? _metrics.OrderBy(sortKey)
            : _metrics.OrderByDescending(sortKey);
        var src = sorted
            .Select(m => new ClassificationMetrics
            {
                Rank = ++rank,
                EpName = m.EpName,
                Mode = m.Mode,
                RawSessionMs = m.RawSessionMs,
                SessionTime = m.SessionTime,
                RawCompileMs = m.RawCompileMs,
                CompileTime = m.CompileTime,
                RawPreprocessMs = m.RawPreprocessMs,
                PreprocessTime = m.PreprocessTime,
                RawInferenceMs = m.RawInferenceMs,
                InferenceTime = m.InferenceTime,
                RawEpPerfMs = m.RawEpPerfMs,
                EpPerfTime = m.EpPerfTime,
                RawTotalMs = m.RawTotalMs,
                TotalTime = m.TotalTime,
                RawMemDeltaMb = m.RawMemDeltaMb,
                MemoryDelta = m.MemoryDelta,
                TopPrediction = m.TopPrediction,
                TopConfidence = m.TopConfidence
            })
            .ToList();
        _metrics.Clear();
        foreach (var item in src)
            _metrics.Add(item);
    }

    private static readonly string[] HeatEmojis = [
        "\U0001F7E2", // green - fastest
        "\U0001F535", // blue
        "\U0001F7E1", // yellow
        "\U0001F7E0", // orange
        "\U0001F534"  // red - slowest
    ];

    private void ApplyHeatMapToColumn(
        Func<ClassificationMetrics, double> getRaw,
        Action<ClassificationMetrics, string> setText,
        Func<ClassificationMetrics, bool> isValid)
    {
        var measured = _metrics.Where(isValid).ToList();
        if (measured.Count == 0) return;

        // Sort ascending (fastest first)
        var ranked = measured.OrderBy(m => getRaw(m)).ToList();
        int n = ranked.Count;

        // Build emoji assignment: 5 buckets [green, blue, yellow, orange, red]
        // Distribute n items across 5 emoji slots using round-robin-ish assignment
        for (int i = 0; i < n; i++)
        {
            int emojiIndex;
            if (n == 1)
            {
                emojiIndex = 0; // single item → green
            }
            else if (n <= 5)
            {
                // n items, n emojis: direct mapping
                // Spread evenly: index 0→green, last→red, interpolate between
                emojiIndex = (int)Math.Round(i * 4.0 / (n - 1));
            }
            else
            {
                // More than 5 items: first=green(0), last=red(4), distribute rest across 1-3
                if (i == 0)
                    emojiIndex = 0;
                else if (i == n - 1)
                    emojiIndex = 4;
                else
                    emojiIndex = 1 + (int)Math.Round((i - 1) * 2.0 / (n - 3));
            }
            if (emojiIndex > 4) emojiIndex = 4;

            setText(ranked[i], $"{HeatEmojis[emojiIndex]} {getRaw(ranked[i]):F1} ms");
        }
    }

    // ============================================================
    //                     EMBEDDINGS TAB
    // ============================================================

    private string EmbModelPath => Path.Combine(AppContext.BaseDirectory, "Assets", TextEmbedder.DefaultModelFileName);
    private string EmbVocabPath => Path.Combine(AppContext.BaseDirectory, "Assets", TextEmbedder.DefaultVocabFileName);
    private string EmbCorpusPath => Path.Combine(AppContext.BaseDirectory, "Assets", "embeddings_corpus.txt");

    private void InitializeEmbeddingsTabUi()
    {
        // Populate the corpus list view from the file even before the embedder loads.
        if (File.Exists(EmbCorpusPath))
        {
            var corpus = File.ReadAllLines(EmbCorpusPath)
                .Select(l => l.Trim())
                .Where(l => !string.IsNullOrEmpty(l))
                .ToList();
            EmbCorpusListView.ItemsSource = corpus;
        }
    }

    /// <summary>
    /// Reuse the EP list already discovered by the Classification tab.
    /// EPs are global to the ORT environment, so we don't need to wait for
    /// the embedding model to be downloaded before showing them.
    /// </summary>
    private void PopulateEmbDevicesFromShared()
    {
        if (_devices.Count == 0) return;
        _embDevices = _devices;
        EmbEpSelector.ItemsSource = _embDevices;
        if (EmbEpSelector.SelectedIndex < 0 && _embDevices.Count > 0)
            EmbEpSelector.SelectedIndex = 0;
    }

    /// <summary>
    /// When the user switches to the Text Similarity tab for the first time,
    /// kick off the model download automatically if it hasn't been done yet.
    /// </summary>
    private void OnMainPivotSelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (sender is not Pivot pivot || pivot.SelectedItem is not PivotItem item) return;
        if (item.Header?.ToString() != "Text Similarity") return;

        if (_embAutoDownloadAttempted) return;
        if (File.Exists(EmbModelPath) && File.Exists(EmbVocabPath)) return; // already downloaded
        _embAutoDownloadAttempted = true;

        // Kick off download in the background (fire-and-forget — UI updates via status text).
        _ = AutoDownloadEmbeddingModelAsync();
    }

    private async Task AutoDownloadEmbeddingModelAsync()
    {
        EmbDownloadModelButton.IsEnabled = false;
        EmbStatusText.Text = "Auto-downloading embedding model (~90 MB)...";
        try
        {
            var progress = new Progress<(string stage, double progress)>(report =>
            {
                EmbStatusText.Text = $"{report.stage} {(int)(report.progress * 100)}%";
            });
            await TextEmbedder.DownloadDefaultArtifactsAsync(EmbModelPath, EmbVocabPath, progress);
            EmbStatusText.Text = "Download complete. Loading model...";
            await TryInitializeEmbedderAsync();
        }
        catch (Exception ex)
        {
            EmbStatusText.Text = $"Auto-download failed: {ex.Message}. Click 'Download Model' to retry.";
            _embAutoDownloadAttempted = false; // allow retry on next tab switch
        }
        finally
        {
            EmbDownloadModelButton.IsEnabled = true;
        }
    }

    private async Task TryInitializeEmbedderAsync()
    {
        if (!File.Exists(EmbModelPath) || !File.Exists(EmbVocabPath))
        {
            EmbStatusText.Text = "Click 'Download Model' to fetch all-MiniLM-L6-v2 (~90 MB).";
            EmbModelNameText.Text = $"{TextEmbedder.DefaultModelFileName} (not downloaded)";
            return;
        }

        EmbStatusText.Text = "Loading embedding model...";
        try
        {
            _embedder?.Dispose();
            _embedder = new TextEmbedder(EmbModelPath, EmbVocabPath, EmbCorpusPath);
            await _embedder.InitializeAsync();
            _embDevices = _embedder.GetAvailableDevices();
            EmbEpSelector.ItemsSource = _embDevices;
            if (_embDevices.Count > 0) EmbEpSelector.SelectedIndex = 0;

            UpdateEmbModelNameDisplay(EmbModelPath);
            PopulateEmbModelInfo();
            EmbSearchButton.IsEnabled = true;
            EmbRunAllButton.IsEnabled = true;
            EmbStatusText.Text = $"Ready — {_embDevices.Count} EP device(s) available, {_embedder.Corpus.Count} corpus sentences embedded.";
        }
        catch (Exception ex)
        {
            EmbStatusText.Text = $"Init error: {ex.Message}";
        }
    }

    private void UpdateEmbModelNameDisplay(string modelPath)
    {
        var fi = new FileInfo(modelPath);
        var sizeMb = fi.Length / (1024.0 * 1024.0);
        var sizeText = sizeMb >= 1024 ? $"{sizeMb / 1024.0:F1} GB" : $"{sizeMb:F1} MB";
        EmbModelNameText.Text = $"{fi.Name} ({sizeText})";
    }

    private async void OnEmbDownloadModel(object sender, RoutedEventArgs e)
    {
        EmbDownloadModelButton.IsEnabled = false;
        EmbStatusText.Text = "Starting download...";
        try
        {
            var progress = new Progress<(string stage, double progress)>(report =>
            {
                EmbStatusText.Text = $"{report.stage} {(int)(report.progress * 100)}%";
            });
            await TextEmbedder.DownloadDefaultArtifactsAsync(EmbModelPath, EmbVocabPath, progress);
            EmbStatusText.Text = "Download complete. Loading model...";
            await TryInitializeEmbedderAsync();
        }
        catch (Exception ex)
        {
            EmbStatusText.Text = $"Download failed: {ex.Message}";
        }
        finally
        {
            EmbDownloadModelButton.IsEnabled = true;
        }
    }

    private async void OnEmbChangeModel(object sender, RoutedEventArgs e)
    {
        var picker = new FileOpenPicker();
        picker.FileTypeFilter.Add(".onnx");
        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(this);
        WinRT.Interop.InitializeWithWindow.Initialize(picker, hwnd);
        var file = await picker.PickSingleFileAsync();
        if (file is null) return;

        // Need a vocab too — ask user
        var vocabPicker = new FileOpenPicker();
        vocabPicker.FileTypeFilter.Add(".txt");
        WinRT.Interop.InitializeWithWindow.Initialize(vocabPicker, hwnd);
        EmbStatusText.Text = "Select the matching vocab.txt for this model...";
        var vocabFile = await vocabPicker.PickSingleFileAsync();
        if (vocabFile is null)
        {
            EmbStatusText.Text = "Cancelled — vocab.txt is required.";
            return;
        }

        _embedder?.Dispose();
        _embedder = new TextEmbedder(file.Path, vocabFile.Path, EmbCorpusPath);
        EmbStatusText.Text = $"Loading {file.Name}...";
        try
        {
            await _embedder.InitializeAsync();
            _embDevices = _embedder.GetAvailableDevices();
            EmbEpSelector.ItemsSource = _embDevices;
            if (_embDevices.Count > 0) EmbEpSelector.SelectedIndex = 0;
            UpdateEmbModelNameDisplay(file.Path);
            PopulateEmbModelInfo();
            _embMatches.Clear();
            _embMetrics.Clear();
            EmbSearchButton.IsEnabled = true;
            EmbRunAllButton.IsEnabled = true;
            EmbStatusText.Text = $"Ready — {_embDevices.Count} EP device(s).";
        }
        catch (Exception ex)
        {
            EmbStatusText.Text = $"Load error: {ex.Message}";
        }
    }

    private async void OnEmbSearch(object sender, RoutedEventArgs e)
    {
        if (_embedder is null || EmbEpSelector.SelectedItem is not EpDeviceInfo device) return;
        var query = QueryTextBox.Text?.Trim();
        if (string.IsNullOrEmpty(query))
        {
            EmbStatusText.Text = "Type a query first.";
            return;
        }

        SetEmbButtonsEnabled(false);
        EmbStatusText.Text = $"Searching with {device.DisplayName}...";

        try
        {
            // Remove existing rows for this EP so cold/warm shows fresh.
            var existing = _embMetrics.Where(m => m.EpName == device.DisplayName).ToList();
            foreach (var item in existing) _embMetrics.Remove(item);

            // JIT cold + warm
            for (int pass = 0; pass < 2; pass++)
            {
                EmbStatusText.Text = $"Searching with {device.DisplayName} (JIT, {(pass == 0 ? "cold" : "warm")})...";
                var result = await _embedder.SearchAsync(query, device, compiled: false);
                _embMetrics.Add(result.Metrics);
                if (pass == 0)
                {
                    _embMatches.Clear();
                    foreach (var m in result.Matches) _embMatches.Add(m);
                }
            }

            // AOT cold + warm
            for (int pass = 0; pass < 2; pass++)
            {
                EmbStatusText.Text = $"Compiling (AOT) & searching with {device.DisplayName} ({(pass == 0 ? "cold" : "warm")})...";
                try
                {
                    var compiled = await _embedder.SearchAsync(query, device, compiled: true);
                    _embMetrics.Add(compiled.Metrics);
                }
                catch (Exception cex)
                {
                    _embMetrics.Add(new EmbeddingMetrics
                    {
                        EpName = device.DisplayName,
                        Mode = pass == 0 ? "AOT (Cold)" : "Warm (AOT-built)",
                        CompileTime = "Error",
                        InferenceTime = "—",
                        TotalTime = ExtractErrorCode(cex)
                    });
                    break;
                }
            }

            ApplyEmbHeatMap();
            EmbStatusText.Text = $"Done — {device.DisplayName} benchmarked (4 runs).";
        }
        catch (Exception ex)
        {
            EmbStatusText.Text = $"Error: {ex.Message}";
        }
        finally
        {
            SetEmbButtonsEnabled(true);
        }
    }

    private async void OnEmbRunAll(object sender, RoutedEventArgs e)
    {
        if (_embedder is null) return;
        var query = QueryTextBox.Text?.Trim();
        if (string.IsNullOrEmpty(query))
        {
            EmbStatusText.Text = "Type a query first.";
            return;
        }

        SetEmbButtonsEnabled(false);
        _embMetrics.Clear();
        // Clear session cache so every EP starts with a true cold run.
        _embedder.ClearSessionCache();
        // 4 runs per EP: JIT cold, JIT warm, AOT cold, AOT warm
        int total = _embDevices.Count * 4;
        int step = 0;

        for (int i = 0; i < _embDevices.Count; i++)
        {
            var device = _embDevices[i];

            // JIT — cold + warm
            for (int pass = 0; pass < 2; pass++)
            {
                step++;
                var passLabel = pass == 0 ? "cold" : "warm";
                EmbStatusText.Text = $"[{step}/{total}] {device.DisplayName} (JIT, {passLabel})...";
                try
                {
                    var result = await _embedder.SearchAsync(query, device, compiled: false);
                    _embMetrics.Add(result.Metrics);
                    ApplyEmbHeatMap();

                    _embMatches.Clear();
                    foreach (var m in result.Matches) _embMatches.Add(m);
                }
                catch (Exception ex)
                {
                    _embMetrics.Add(new EmbeddingMetrics
                    {
                        EpName = device.DisplayName,
                        Mode = pass == 0 ? "JIT (Cold)" : "Warm (JIT-built)",
                        InferenceTime = "Error",
                        TotalTime = ExtractErrorCode(ex)
                    });
                    ApplyEmbHeatMap();
                }
            }

            // AOT — cold + warm
            for (int pass = 0; pass < 2; pass++)
            {
                step++;
                var passLabel = pass == 0 ? "cold" : "warm";
                EmbStatusText.Text = $"[{step}/{total}] {device.DisplayName} (AOT, {passLabel})...";
                try
                {
                    var compiled = await _embedder.SearchAsync(query, device, compiled: true);
                    _embMetrics.Add(compiled.Metrics);
                    ApplyEmbHeatMap();
                }
                catch (Exception ex)
                {
                    _embMetrics.Add(new EmbeddingMetrics
                    {
                        EpName = device.DisplayName,
                        Mode = pass == 0 ? "AOT (Cold)" : "Warm (AOT-built)",
                        CompileTime = "Error",
                        InferenceTime = "—",
                        TotalTime = ExtractErrorCode(ex)
                    });
                    ApplyEmbHeatMap();
                }
            }
        }

        EmbStatusText.Text = $"Comparison complete — {_embDevices.Count} EP(s) x 4 runs benchmarked.";
        SetEmbButtonsEnabled(true);
    }

    private void SetEmbButtonsEnabled(bool enabled)
    {
        EmbSearchButton.IsEnabled = enabled && _embedder != null;
        EmbRunAllButton.IsEnabled = enabled && _embedder != null;
        EmbExportButton.IsEnabled = _embMetrics.Count > 0;
    }

    private async void OnEmbDownloadCorpus(object sender, RoutedEventArgs e)
    {
        if (!File.Exists(EmbCorpusPath))
        {
            EmbStatusText.Text = "Corpus file not found.";
            return;
        }

        var picker = new Windows.Storage.Pickers.FileSavePicker();
        picker.SuggestedStartLocation = Windows.Storage.Pickers.PickerLocationId.DocumentsLibrary;
        picker.FileTypeChoices.Add("Text", new List<string> { ".txt" });
        picker.SuggestedFileName = "coffee_shop_knowledge_bank";

        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(this);
        WinRT.Interop.InitializeWithWindow.Initialize(picker, hwnd);

        var file = await picker.PickSaveFileAsync();
        if (file is null) return;

        try
        {
            var contents = await File.ReadAllTextAsync(EmbCorpusPath);
            await Windows.Storage.FileIO.WriteTextAsync(file, contents);
            EmbStatusText.Text = $"Corpus saved to {file.Name}";
        }
        catch (Exception ex)
        {
            EmbStatusText.Text = $"Save failed: {ex.Message}";
        }
    }

    private async void OnEmbExportCsv(object sender, RoutedEventArgs e)
    {
        if (_embMetrics.Count == 0) return;
        var picker = new Windows.Storage.Pickers.FileSavePicker();
        picker.SuggestedStartLocation = Windows.Storage.Pickers.PickerLocationId.DocumentsLibrary;
        picker.FileTypeChoices.Add("CSV", new List<string> { ".csv" });
        picker.SuggestedFileName = $"EP_Embeddings_Benchmark_{DateTime.Now:yyyyMMdd_HHmmss}";
        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(this);
        WinRT.Interop.InitializeWithWindow.Initialize(picker, hwnd);
        var file = await picker.PickSaveFileAsync();
        if (file is null) return;

        var lines = new List<string>
        {
            "Rank,Execution Provider,Mode,Tokenization (ms),Session Creation (ms),AOT Compile (ms),Inference (ms),EP Total (ms),Total (ms),Memory Delta (MB),Top Match,Similarity"
        };
        foreach (var m in _embMetrics)
        {
            var tok = m.RawTokenizeMs >= 0 ? $"{m.RawTokenizeMs:F1}" : "";
            var sess = m.RawSessionMs >= 0 ? $"{m.RawSessionMs:F1}" : "";
            var comp = m.RawCompileMs >= 0 ? $"{m.RawCompileMs:F1}" : m.CompileTime.Replace("—", "").Trim();
            var inf = m.RawInferenceMs >= 0 ? $"{m.RawInferenceMs:F1}" : "Error";
            var ep = m.RawEpPerfMs >= 0 ? $"{m.RawEpPerfMs:F1}" : "Error";
            var tot = m.RawTotalMs >= 0 ? $"{m.RawTotalMs:F1}" : "Error";
            var memory = double.IsNaN(m.RawMemDeltaMb) ? "" : $"{m.RawMemDeltaMb:F1}";
            lines.Add($"{m.Rank},\"{m.EpName}\",{m.Mode},{tok},{sess},{comp},{inf},{ep},{tot},{memory},\"{m.TopMatch}\",{m.TopSimilarity}");
        }
        await Windows.Storage.FileIO.WriteLinesAsync(file, lines);
        EmbStatusText.Text = $"Exported {_embMetrics.Count} rows to {file.Name}";
    }

    private void OnEmbColumnHeaderClick(object sender, RoutedEventArgs e)
    {
        if (sender is not Button btn || btn.Tag is not string col) return;
        if (_embMetrics.Count == 0) return;

        if (_embSortColumn == col)
            _embSortAscending = !_embSortAscending;
        else
        {
            _embSortColumn = col;
            _embSortAscending = true;
        }

        UpdateEmbColumnHeaderArrows();
        ApplyEmbHeatMap();
    }

    private void UpdateEmbColumnHeaderArrows()
    {
        var arrow = _embSortAscending ? " ▲" : " ▼";
        var headers = new Dictionary<string, TextBlock>
        {
            ["EpName"] = EmbHeader_EpName,
            ["Mode"] = EmbHeader_Mode,
            ["Tokenize"] = EmbHeader_Tokenize,
            ["Session"] = EmbHeader_Session,
            ["Compile"] = EmbHeader_Compile,
            ["Inference"] = EmbHeader_Inference,
            ["EpLatency"] = EmbHeader_EpLatency,
            ["Total"] = EmbHeader_Total,
            ["Memory"] = EmbHeader_Memory,
            ["TopMatch"] = EmbHeader_TopMatch,
            ["Similarity"] = EmbHeader_Similarity
        };
        foreach (var (key, tb) in headers)
            tb.Text = key == _embSortColumn ? EmbColumnLabels[key] + arrow : EmbColumnLabels[key];
    }

    private void ApplyEmbHeatMap()
    {
        ApplyEmbHeatMapToColumn(m => m.RawInferenceMs, (m, t) => m.InferenceTime = t, m => m.RawInferenceMs >= 0);
        ApplyEmbHeatMapToColumn(m => m.RawEpPerfMs, (m, t) => m.EpPerfTime = t, m => m.RawEpPerfMs >= 0);

        Func<EmbeddingMetrics, object> sortKey = _embSortColumn switch
        {
            "EpName" => m => m.EpName,
            "Mode" => m => m.Mode,
            "Tokenize" => m => m.RawTokenizeMs < 0 ? (object)double.MaxValue : m.RawTokenizeMs,
            "Session" => m => m.RawSessionMs < 0 ? (object)double.MaxValue : m.RawSessionMs,
            "Compile" => m => m.RawCompileMs < 0 ? (object)double.MaxValue : m.RawCompileMs,
            "Inference" => m => m.RawInferenceMs < 0 ? (object)double.MaxValue : m.RawInferenceMs,
            "EpLatency" => m => m.RawEpPerfMs < 0 ? (object)double.MaxValue : m.RawEpPerfMs,
            "Total" => m => m.RawTotalMs < 0 ? (object)double.MaxValue : m.RawTotalMs,
            "Memory" => m => double.IsNaN(m.RawMemDeltaMb) ? (object)double.MaxValue : m.RawMemDeltaMb,
            "TopMatch" => m => m.TopMatch,
            "Similarity" => m => m.TopSimilarity,
            _ => m => m.RawEpPerfMs < 0 ? (object)double.MaxValue : m.RawEpPerfMs
        };

        int rank = 0;
        var sorted = _embSortAscending ? _embMetrics.OrderBy(sortKey) : _embMetrics.OrderByDescending(sortKey);
        var src = sorted.Select(m => new EmbeddingMetrics
        {
            Rank = ++rank,
            EpName = m.EpName,
            Mode = m.Mode,
            RawTokenizeMs = m.RawTokenizeMs,
            TokenizeTime = m.TokenizeTime,
            RawSessionMs = m.RawSessionMs,
            SessionTime = m.SessionTime,
            RawCompileMs = m.RawCompileMs,
            CompileTime = m.CompileTime,
            RawInferenceMs = m.RawInferenceMs,
            InferenceTime = m.InferenceTime,
            RawEpPerfMs = m.RawEpPerfMs,
            EpPerfTime = m.EpPerfTime,
            RawTotalMs = m.RawTotalMs,
            TotalTime = m.TotalTime,
            RawMemDeltaMb = m.RawMemDeltaMb,
            MemoryDelta = m.MemoryDelta,
            TopMatch = m.TopMatch,
            TopSimilarity = m.TopSimilarity
        }).ToList();
        _embMetrics.Clear();
        foreach (var item in src) _embMetrics.Add(item);
    }

    private void ApplyEmbHeatMapToColumn(
        Func<EmbeddingMetrics, double> getRaw,
        Action<EmbeddingMetrics, string> setText,
        Func<EmbeddingMetrics, bool> isValid)
    {
        var measured = _embMetrics.Where(isValid).ToList();
        if (measured.Count == 0) return;
        var ranked = measured.OrderBy(m => getRaw(m)).ToList();
        int n = ranked.Count;
        for (int i = 0; i < n; i++)
        {
            int emojiIndex;
            if (n == 1) emojiIndex = 0;
            else if (n <= 5) emojiIndex = (int)Math.Round(i * 4.0 / (n - 1));
            else
            {
                if (i == 0) emojiIndex = 0;
                else if (i == n - 1) emojiIndex = 4;
                else emojiIndex = 1 + (int)Math.Round((i - 1) * 2.0 / (n - 3));
            }
            if (emojiIndex > 4) emojiIndex = 4;
            setText(ranked[i], $"{HeatEmojis[emojiIndex]} {getRaw(ranked[i]):F1} ms");
        }
    }
}
