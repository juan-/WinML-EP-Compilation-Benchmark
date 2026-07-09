using System.ComponentModel;
using System.Diagnostics;
using System.Text.Json;

namespace WinMLResNet;

/// <summary>A model file the user wants to include in the compilation sweep.</summary>
public class CompileModelItem
{
    public string Label { get; set; } = "";
    public string Path { get; set; } = "";
    public long SizeBytes { get; set; }
    public bool Include { get; set; } = true;
    public string SizeText
    {
        get
        {
            var mb = SizeBytes / (1024.0 * 1024.0);
            return mb >= 1024 ? $"{mb / 1024.0:F1} GB" : $"{mb:F1} MB";
        }
    }
}

/// <summary>An execution-provider device that can be selected for the sweep.</summary>
public class CompileEpItem
{
    public string EpName { get; set; } = "";
    public string DeviceType { get; set; } = "";
    public string DisplayName { get; set; } = "";
    public string Note { get; set; } = "";
    public bool Include { get; set; }
}

/// <summary>A read-only diagnostics row explaining an execution provider's availability:
/// whether its library registered, and whether it enumerated a selectable device.</summary>
public class EpDiagnosticRow
{
    public string Provider { get; set; } = "";
    public string Status { get; set; } = "";
    public string Detail { get; set; } = "";
}

/// <summary>One row of the compilation results grid. Mutable cells raise change notifications
/// so a row can be shown as "running" and then updated in place when the worker finishes.</summary>
public class CompileResultRow : INotifyPropertyChanged
{
    public int Rank { get; set; }
    public string ModelLabel { get; set; } = "";
    public string EpName { get; set; } = "";
    public string DeviceType { get; set; } = "";
    public double RawCompileMs { get; set; } = -1;
    public long CompiledBytes { get; set; } = -1;

    private string _compileText = "";
    public string CompileText { get => _compileText; set { _compileText = value; Raise(nameof(CompileText)); } }

    private string _sizeText = "";
    public string SizeText { get => _sizeText; set { _sizeText = value; Raise(nameof(SizeText)); } }

    private string _status = "";
    public string Status { get => _status; set { _status = value; Raise(nameof(Status)); } }

    public string? Error { get; set; }

    public event PropertyChangedEventHandler? PropertyChanged;
    private void Raise(string name) => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}

/// <summary>Outcome of a single isolated compile job.</summary>
public class CompileRunResult
{
    public bool Ok { get; set; }
    public bool Crashed { get; set; }
    public double CompileMs { get; set; } = -1;
    public long CompiledBytes { get; set; } = -1;
    public string? Error { get; set; }
}

/// <summary>
/// Runs each model+EP compilation in the isolated worker process (the CompileErrorTest exe) so a
/// native crash from a provider without matching hardware can't bring down the UI.
/// </summary>
public static class CompileBenchmarkRunner
{
    private class WorkerJson
    {
        public string? label { get; set; }
        public string? model { get; set; }
        public string? ep { get; set; }
        public string? deviceType { get; set; }
        public double compileMs { get; set; }
        public long compiledBytes { get; set; }
        public bool ok { get; set; }
        public string? error { get; set; }
    }

    /// <summary>Finds the worker exe next to the app, or in the sibling CompileErrorTest build output.</summary>
    public static string? LocateWorker()
    {
        var appBase = AppContext.BaseDirectory.TrimEnd(System.IO.Path.DirectorySeparatorChar);

        // 1. Copied next to the app.
        var direct = System.IO.Path.Combine(appBase, "CompileErrorTest.exe");
        if (File.Exists(direct)) return direct;

        // 2. Sibling project output: <root>\CompileErrorTest\bin\<Platform>\<Config>\<TFM>\CompileErrorTest.exe
        var appDir = new DirectoryInfo(appBase);
        var tfm = appDir.Name;                          // net8.0-windows10.0.22621.0
        var config = appDir.Parent?.Name;               // Debug / Release
        var platform = appDir.Parent?.Parent?.Name;     // x64 / ARM64
        var projRoot = appDir.Parent?.Parent?.Parent?.Parent; // project root (…\bin\.. up)

        if (projRoot != null)
        {
            var cand = System.IO.Path.Combine(projRoot.FullName, "CompileErrorTest", "bin",
                platform ?? "", config ?? "", tfm, "CompileErrorTest.exe");
            if (File.Exists(cand)) return cand;

            var workerBin = System.IO.Path.Combine(projRoot.FullName, "CompileErrorTest", "bin");
            if (Directory.Exists(workerBin))
            {
                var found = Directory.GetFiles(workerBin, "CompileErrorTest.exe", SearchOption.AllDirectories);
                var best = found.FirstOrDefault(f => f.Contains($"{System.IO.Path.DirectorySeparatorChar}{platform}{System.IO.Path.DirectorySeparatorChar}{config}{System.IO.Path.DirectorySeparatorChar}"))
                           ?? found.FirstOrDefault();
                if (best != null) return best;
            }
        }

        return null;
    }

    public static async Task<CompileRunResult> RunAsync(
        string workerExe, string modelPath, string epName, string deviceType, string label,
        int timeoutMs = 180000)
    {
        var outFile = System.IO.Path.Combine(System.IO.Path.GetTempPath(),
            $"epcompile-{Guid.NewGuid():N}.json");

        var psi = new ProcessStartInfo
        {
            FileName = workerExe,
            WorkingDirectory = System.IO.Path.GetDirectoryName(workerExe),
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        psi.ArgumentList.Add("--compile");
        psi.ArgumentList.Add("--model"); psi.ArgumentList.Add(modelPath);
        psi.ArgumentList.Add("--ep"); psi.ArgumentList.Add(epName);
        psi.ArgumentList.Add("--deviceType"); psi.ArgumentList.Add(deviceType);
        psi.ArgumentList.Add("--out"); psi.ArgumentList.Add(outFile);
        psi.ArgumentList.Add("--label"); psi.ArgumentList.Add(label);

        try
        {
            using var proc = Process.Start(psi);
            if (proc is null)
                return new CompileRunResult { Ok = false, Crashed = true, Error = "Failed to start worker process." };

            // Drain output streams so the child doesn't block on a full pipe.
            var stdoutTask = proc.StandardOutput.ReadToEndAsync();
            var stderrTask = proc.StandardError.ReadToEndAsync();

            using var cts = new CancellationTokenSource(timeoutMs);
            try
            {
                await proc.WaitForExitAsync(cts.Token);
            }
            catch (OperationCanceledException)
            {
                try { proc.Kill(true); } catch { }
                return new CompileRunResult { Ok = false, Crashed = true, Error = $"Timed out after {timeoutMs / 1000}s." };
            }

            await Task.WhenAny(Task.WhenAll(stdoutTask, stderrTask), Task.Delay(2000));

            if (File.Exists(outFile))
            {
                try
                {
                    var json = await File.ReadAllTextAsync(outFile);
                    var dto = JsonSerializer.Deserialize<WorkerJson>(json,
                        new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                    if (dto != null)
                    {
                        return new CompileRunResult
                        {
                            Ok = dto.ok,
                            Crashed = false,
                            CompileMs = dto.compileMs,
                            CompiledBytes = dto.compiledBytes,
                            Error = dto.error
                        };
                    }
                }
                catch (Exception ex)
                {
                    return new CompileRunResult { Ok = false, Crashed = false, Error = $"Bad worker output: {ex.Message}" };
                }
                finally
                {
                    try { File.Delete(outFile); } catch { }
                }
            }

            // No result file → the worker died before it could report (native crash).
            var code = proc.ExitCode;
            return new CompileRunResult
            {
                Ok = false,
                Crashed = true,
                Error = $"Worker exited (0x{code & 0xFFFFFFFF:X8}) without a result — likely a native crash."
            };
        }
        catch (Exception ex)
        {
            return new CompileRunResult { Ok = false, Crashed = true, Error = ex.Message };
        }
        finally
        {
            try { if (File.Exists(outFile)) File.Delete(outFile); } catch { }
        }
    }
}
