using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Windows.AI.MachineLearning;
using Microsoft.ML.Tokenizers;
using System.Diagnostics;
using System.Net.Http;

namespace WinMLResNet;

public class EmbeddingMatch
{
    public int Rank { get; set; }
    public string Text { get; set; } = "";
    public string Score { get; set; } = "";
    public double RawSimilarity { get; set; }
}

public class EmbeddingMetrics
{
    public int Rank { get; set; }
    public string EpName { get; set; } = "";
    public string Mode { get; set; } = "JIT";
    public double RawSessionMs { get; set; } = -1;
    public string SessionTime { get; set; } = "";
    public double RawCompileMs { get; set; } = -1;
    public string CompileTime { get; set; } = "—";
    public double RawTokenizeMs { get; set; } = -1;
    public string TokenizeTime { get; set; } = "";
    public double RawInferenceMs { get; set; } = -1;
    public string InferenceTime { get; set; } = "";
    public double RawEpPerfMs { get; set; } = -1;
    public string EpPerfTime { get; set; } = "";
    public double RawTotalMs { get; set; } = -1;
    public string TotalTime { get; set; } = "";
    public double RawMemDeltaMb { get; set; } = double.NaN;
    public string MemoryDelta { get; set; } = "";
    public string TopMatch { get; set; } = "";
    public string TopSimilarity { get; set; } = "";
}

public class EmbeddingSearchResult
{
    public List<EmbeddingMatch> Matches { get; set; } = new();
    public EmbeddingMetrics Metrics { get; set; } = new();
}

public class TextEmbedder : IDisposable
{
    // Default model: sentence-transformers/all-MiniLM-L6-v2 (~90 MB, 22M params, 384-dim)
    public const string DefaultModelUrl = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx";
    public const string DefaultVocabUrl = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt";
    public const string DefaultModelFileName = "all-MiniLM-L6-v2.onnx";
    public const string DefaultVocabFileName = "all-MiniLM-L6-v2-vocab.txt";

    private readonly string _modelPath;
    private readonly string _vocabPath;
    private BertTokenizer? _tokenizer;
    private bool _initialized;
    private readonly Dictionary<string, InferenceSession> _sessions = new();

    // Corpus + their precomputed embeddings (computed once on CPU at init time)
    private readonly List<string> _corpus = new();
    private float[][]? _corpusEmbeddings;

    public IReadOnlyList<string> Corpus => _corpus;

    public TextEmbedder(string modelPath, string vocabPath, string corpusPath)
    {
        _modelPath = modelPath;
        _vocabPath = vocabPath;

        if (File.Exists(corpusPath))
        {
            foreach (var line in File.ReadAllLines(corpusPath))
            {
                var trimmed = line.Trim();
                if (!string.IsNullOrEmpty(trimmed))
                    _corpus.Add(trimmed);
            }
        }

        // ORT env may already be initialized by ImageClassifier; that's fine — only first wins.
        try
        {
            var envOptions = new EnvironmentCreationOptions
            {
                logId = "WinMLResNet",
                logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
            };
            OrtEnv.CreateInstanceWithOptions(ref envOptions);
        }
        catch { /* already initialized — ignore */ }
    }

    public bool ArtifactsExist => File.Exists(_modelPath) && File.Exists(_vocabPath);

    /// <summary>
    /// Downloads model + vocab from HuggingFace into Assets/. Reports progress 0..1.
    /// </summary>
    public static async Task DownloadDefaultArtifactsAsync(
        string modelTargetPath,
        string vocabTargetPath,
        IProgress<(string stage, double progress)>? progress = null,
        CancellationToken ct = default)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(modelTargetPath)!);

        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
        http.DefaultRequestHeaders.UserAgent.ParseAdd("WinML-EP-Benchmarking-Sample-App/1.0");

        // Vocab first (small, ~230 KB)
        progress?.Report(("Downloading vocab.txt...", 0));
        await DownloadFileAsync(http, DefaultVocabUrl, vocabTargetPath, p => progress?.Report(("Downloading vocab.txt...", p * 0.05)), ct);

        // Model (~90 MB)
        progress?.Report(("Downloading model.onnx...", 0.05));
        await DownloadFileAsync(http, DefaultModelUrl, modelTargetPath, p => progress?.Report(("Downloading model.onnx...", 0.05 + p * 0.95)), ct);

        progress?.Report(("Done", 1.0));
    }

    private static async Task DownloadFileAsync(HttpClient http, string url, string targetPath, Action<double> onProgress, CancellationToken ct)
    {
        using var response = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, ct);
        response.EnsureSuccessStatusCode();
        var totalBytes = response.Content.Headers.ContentLength ?? -1L;

        var tmp = targetPath + ".tmp";
        using (var src = await response.Content.ReadAsStreamAsync(ct))
        using (var dst = File.Create(tmp))
        {
            var buffer = new byte[81920];
            long readSoFar = 0;
            int read;
            while ((read = await src.ReadAsync(buffer, ct)) > 0)
            {
                await dst.WriteAsync(buffer.AsMemory(0, read), ct);
                readSoFar += read;
                if (totalBytes > 0)
                    onProgress(readSoFar / (double)totalBytes);
            }
        }
        if (File.Exists(targetPath)) File.Delete(targetPath);
        File.Move(tmp, targetPath);
    }

    public async Task InitializeAsync()
    {
        if (_initialized) return;

        if (!ArtifactsExist)
            throw new FileNotFoundException("Embedding model or vocab file is missing. Click 'Download Model' first.");

        // Register only the EPs a crash-isolated worker confirmed produce a device (shared approach
        // with ImageClassifier — avoids force-loading fragile vendor libraries in-process).
        await EpInventory.EnsureAsync();
        var safe = EpInventory.DeviceProducingProviders
            .Where(p => !p.Equals("CPUExecutionProvider", StringComparison.OrdinalIgnoreCase)
                     && !p.Equals("DmlExecutionProvider", StringComparison.OrdinalIgnoreCase))
            .ToList();
        if (safe.Count > 0)
            await EpRegistration.RegisterSpecificAsync(safe);

        // Load tokenizer
        using (var vocabStream = File.OpenRead(_vocabPath))
        {
            _tokenizer = BertTokenizer.Create(vocabStream);
        }

        // Pre-compute corpus embeddings on CPU once (so per-EP benchmark only times the query embedding).
        _corpusEmbeddings = await Task.Run(() => ComputeCorpusEmbeddings());

        _initialized = true;
    }

    private float[][] ComputeCorpusEmbeddings()
    {
        // Use a vanilla CPU session for corpus precompute (no EP selection needed)
        using var session = new InferenceSession(_modelPath);
        var embeddings = new float[_corpus.Count][];
        for (int i = 0; i < _corpus.Count; i++)
        {
            embeddings[i] = EmbedSentence(session, _corpus[i]);
        }
        return embeddings;
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

        var quantizedTypes = new[] { "Int8", "UInt8", "Int4", "UInt4", "Float16" };
        var foundQuantized = allDataTypes.Where(d => quantizedTypes.Any(q => d.Contains(q, StringComparison.OrdinalIgnoreCase))).ToList();
        bool isQuantized = foundQuantized.Count > 0;
        string quantDetail = isQuantized ? string.Join(", ", foundQuantized) : "No (Float32)";

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

    private (InferenceSession session, bool wasCached) GetOrCreateSession(EpDeviceInfo epDevice)
    {
        var key = $"{epDevice.DisplayName}|uncompiled";
        if (_sessions.TryGetValue(key, out var cached)) return (cached, true);

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
        if (_sessions.TryGetValue(key, out var cached)) return (cached, 0, true);

        var ortEnv = OrtEnv.Instance();
        var sessionOptions = new SessionOptions();
        sessionOptions.AppendExecutionProvider(ortEnv, new[] { epDevice.Device }, new Dictionary<string, string>());

        var epSafe = epDevice.DisplayName.Replace(" ", "_").Replace("(", "").Replace(")", "");
        var compiledPath = Path.Combine(
            Path.GetDirectoryName(_modelPath)!,
            $"{Path.GetFileNameWithoutExtension(_modelPath)}-compiled-{epSafe}.onnx");
        if (File.Exists(compiledPath)) File.Delete(compiledPath);

        var sw = Stopwatch.StartNew();
        using var compileOptions = new OrtModelCompilationOptions(sessionOptions);
        compileOptions.SetInputModelPath(_modelPath);
        compileOptions.SetOutputModelPath(compiledPath);
        compileOptions.CompileModel();
        sw.Stop();

        var session = new InferenceSession(compiledPath, sessionOptions);
        _sessions[key] = session;
        return (session, sw.Elapsed.TotalMilliseconds, false);
    }

    /// <summary>
    /// Tokenize a sentence into the three int64 input tensors (input_ids, attention_mask, token_type_ids).
    /// </summary>
    private (DenseTensor<long> ids, DenseTensor<long> mask, DenseTensor<long> typeIds, int seqLen) Tokenize(string text)
    {
        if (_tokenizer is null)
            throw new InvalidOperationException("Tokenizer not initialized.");

        var ids = _tokenizer.EncodeToIds(text);
        // EncodeToIds in Microsoft.ML.Tokenizers BertTokenizer returns the IDs WITH special tokens [CLS] ... [SEP].
        // Cap to 256 for safety on long inputs.
        const int maxLen = 256;
        var idArr = ids.Take(maxLen).Select(i => (long)i).ToArray();
        int seqLen = idArr.Length;

        var idsTensor = new DenseTensor<long>(idArr, new[] { 1, seqLen });
        var maskTensor = new DenseTensor<long>(Enumerable.Repeat(1L, seqLen).ToArray(), new[] { 1, seqLen });
        var typeIdsTensor = new DenseTensor<long>(new long[seqLen], new[] { 1, seqLen }); // all zeros

        return (idsTensor, maskTensor, typeIdsTensor, seqLen);
    }

    /// <summary>
    /// Run the model on a tokenized sentence and produce a 384-dim mean-pooled, L2-normalized embedding.
    /// </summary>
    private float[] EmbedSentence(InferenceSession session, string text)
    {
        var (ids, mask, typeIds, seqLen) = Tokenize(text);

        var inputs = new List<NamedOnnxValue>();
        // Match the model's expected input names dynamically
        foreach (var (name, _) in session.InputMetadata)
        {
            if (name.Contains("input_ids", StringComparison.OrdinalIgnoreCase))
                inputs.Add(NamedOnnxValue.CreateFromTensor(name, ids));
            else if (name.Contains("attention_mask", StringComparison.OrdinalIgnoreCase))
                inputs.Add(NamedOnnxValue.CreateFromTensor(name, mask));
            else if (name.Contains("token_type", StringComparison.OrdinalIgnoreCase))
                inputs.Add(NamedOnnxValue.CreateFromTensor(name, typeIds));
        }

        using var results = session.Run(inputs);
        // Prefer last_hidden_state for mean pooling; fall back to first output if name differs.
        var output = results.FirstOrDefault(r => r.Name.Contains("last_hidden_state", StringComparison.OrdinalIgnoreCase))
                     ?? results.First();
        var tensor = output.AsTensor<float>();

        // Shape: [1, seq_len, hidden_dim]
        int hidden = tensor.Dimensions[2];
        var pooled = new float[hidden];
        // Mean-pool over seq_len dim using attention mask (all 1s here so just average all tokens)
        for (int t = 0; t < seqLen; t++)
            for (int h = 0; h < hidden; h++)
                pooled[h] += tensor[0, t, h];
        for (int h = 0; h < hidden; h++)
            pooled[h] /= seqLen;

        // L2 normalize
        double normSq = 0;
        for (int h = 0; h < hidden; h++) normSq += pooled[h] * pooled[h];
        float norm = (float)Math.Sqrt(normSq);
        if (norm > 1e-9f)
            for (int h = 0; h < hidden; h++) pooled[h] /= norm;

        return pooled;
    }

    private static double Cosine(float[] a, float[] b)
    {
        // Both vectors are L2-normalized → cosine similarity = dot product
        double dot = 0;
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++) dot += a[i] * b[i];
        return dot;
    }

    public async Task<EmbeddingSearchResult> SearchAsync(string query, EpDeviceInfo epDevice, bool compiled = false)
    {
        await InitializeAsync();
        if (_corpusEmbeddings is null) throw new InvalidOperationException("Corpus embeddings missing.");

        // Capture working-set baseline before any work for this run.
        var process = Process.GetCurrentProcess();
        process.Refresh();
        long memBefore = process.WorkingSet64;

        var totalSw = Stopwatch.StartNew();

        // Tokenize query (timed)
        var tokSw = Stopwatch.StartNew();
        var (ids, mask, typeIds, seqLen) = await Task.Run(() => Tokenize(query));
        tokSw.Stop();

        // Get/create session (timed)
        var sessionSw = Stopwatch.StartNew();
        InferenceSession session;
        double compileMs = 0;
        bool sessionCached = false;
        if (compiled)
            (session, compileMs, sessionCached) = await Task.Run(() => GetOrCreateCompiledSession(epDevice));
        else
            (session, sessionCached) = await Task.Run(() => GetOrCreateSession(epDevice));
        sessionSw.Stop();
        bool isFirstRun = !sessionCached;

        // Inference (timed)
        var inputs = new List<NamedOnnxValue>();
        foreach (var (name, _) in session.InputMetadata)
        {
            if (name.Contains("input_ids", StringComparison.OrdinalIgnoreCase))
                inputs.Add(NamedOnnxValue.CreateFromTensor(name, ids));
            else if (name.Contains("attention_mask", StringComparison.OrdinalIgnoreCase))
                inputs.Add(NamedOnnxValue.CreateFromTensor(name, mask));
            else if (name.Contains("token_type", StringComparison.OrdinalIgnoreCase))
                inputs.Add(NamedOnnxValue.CreateFromTensor(name, typeIds));
        }

        var infSw = Stopwatch.StartNew();
        using var results = await Task.Run(() => session.Run(inputs));
        infSw.Stop();

        // Capture working-set right after inference (post-processing is negligible)
        process.Refresh();
        long memAfter = process.WorkingSet64;
        double memDeltaMb = (memAfter - memBefore) / (1024.0 * 1024.0);

        // Pool + normalize (this is post-processing, not part of pure inference timing)
        var output = results.FirstOrDefault(r => r.Name.Contains("last_hidden_state", StringComparison.OrdinalIgnoreCase))
                     ?? results.First();
        var tensor = output.AsTensor<float>();
        int hidden = tensor.Dimensions[2];
        var pooled = new float[hidden];
        for (int t = 0; t < seqLen; t++)
            for (int h = 0; h < hidden; h++)
                pooled[h] += tensor[0, t, h];
        for (int h = 0; h < hidden; h++)
            pooled[h] /= seqLen;
        double normSq = 0;
        for (int h = 0; h < hidden; h++) normSq += pooled[h] * pooled[h];
        float norm = (float)Math.Sqrt(normSq);
        if (norm > 1e-9f)
            for (int h = 0; h < hidden; h++) pooled[h] /= norm;

        // Compare with corpus
        var sims = new (string text, double sim)[_corpus.Count];
        for (int i = 0; i < _corpus.Count; i++)
            sims[i] = (_corpus[i], Cosine(pooled, _corpusEmbeddings[i]));

        var top5 = sims.OrderByDescending(s => s.sim).Take(5).ToList();
        var matches = new List<EmbeddingMatch>();
        for (int i = 0; i < top5.Count; i++)
        {
            matches.Add(new EmbeddingMatch
            {
                Rank = i + 1,
                Text = top5[i].text,
                RawSimilarity = top5[i].sim,
                Score = $"{top5[i].sim * 100:F1}%"
            });
        }

        totalSw.Stop();

        return new EmbeddingSearchResult
        {
            Matches = matches,
            Metrics = new EmbeddingMetrics
            {
                EpName = epDevice.DisplayName,
                Mode = isFirstRun
                    ? (compiled ? "AOT (Cold)" : "JIT (Cold)")
                    : (compiled ? "Warm (AOT-built)" : "Warm (JIT-built)"),
                RawTokenizeMs = tokSw.Elapsed.TotalMilliseconds,
                TokenizeTime = $"{tokSw.Elapsed.TotalMilliseconds:F1} ms",
                RawSessionMs = sessionSw.Elapsed.TotalMilliseconds,
                SessionTime = $"{sessionSw.Elapsed.TotalMilliseconds:F1} ms",
                RawCompileMs = compiled && !sessionCached ? compileMs : -1,
                CompileTime = compiled
                    ? (sessionCached ? "AOT cached" : $"{compileMs:F1} ms")
                    : (sessionCached ? "session cached" : "during session creation"),
                RawInferenceMs = infSw.Elapsed.TotalMilliseconds,
                InferenceTime = $"{infSw.Elapsed.TotalMilliseconds:F1} ms",
                RawEpPerfMs = sessionSw.Elapsed.TotalMilliseconds + (compiled && !sessionCached ? compileMs : 0) + infSw.Elapsed.TotalMilliseconds,
                EpPerfTime = $"{(sessionSw.Elapsed.TotalMilliseconds + (compiled && !sessionCached ? compileMs : 0) + infSw.Elapsed.TotalMilliseconds):F1} ms",
                RawTotalMs = totalSw.Elapsed.TotalMilliseconds,
                TotalTime = $"{totalSw.Elapsed.TotalMilliseconds:F1} ms",
                RawMemDeltaMb = memDeltaMb,
                MemoryDelta = ImageClassifier.FormatMemDelta(memDeltaMb, isFirstRun),
                TopMatch = matches.Count > 0 ? matches[0].Text : "",
                TopSimilarity = matches.Count > 0 ? matches[0].Score : ""
            }
        };
    }

    public void Dispose()
    {
        foreach (var s in _sessions.Values) s.Dispose();
        _sessions.Clear();
    }

    /// <summary>
    /// Disposes all cached InferenceSessions so the next SearchAsync call for any EP+Mode
    /// will be a true 'cold' run (model loaded into memory fresh).
    /// </summary>
    public void ClearSessionCache()
    {
        foreach (var s in _sessions.Values) s.Dispose();
        _sessions.Clear();
        GC.Collect();
        GC.WaitForPendingFinalizers();
    }
}
