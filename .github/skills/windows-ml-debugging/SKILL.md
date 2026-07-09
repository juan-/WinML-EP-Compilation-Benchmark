---
name: windows-ml-debugging
description: 'Debug Windows ML and ONNX Runtime version and runtime issues. Use when: checking ORT version, WASDK version at runtime, managed ORT reports 0.0.0, crash logs, ORT logging, troubleshooting Windows ML errors.'
---

# Windows ML Debugging & Version Info

## When to Use
- Need to display or log runtime version info for Windows ML components
- Getting unexpected version numbers (e.g., ORT showing `0.0.0`)
- Troubleshooting runtime errors or crashes
- Setting up diagnostic logging

## Getting Version Info at Runtime

### ONNX Runtime Version
The managed `Microsoft.ML.OnnxRuntime.dll` reports `0.0.0+<hash>`. Read the native DLL instead:

```csharp
var ortDllPath = Path.Combine(AppContext.BaseDirectory, "onnxruntime.dll");
var versionInfo = System.Diagnostics.FileVersionInfo.GetVersionInfo(ortDllPath);
// ProductVersion: "1.23.20260219.1.4e0442d" — extract major.minor
var parts = versionInfo.ProductVersion.Split('.');
string ortVersion = $"{parts[0]}.{parts[1]}"; // "1.23"
```

### Windows ML Version
```csharp
var mlInfoVer = typeof(ExecutionProviderCatalog).Assembly
    .GetCustomAttribute<AssemblyInformationalVersionAttribute>()
    ?.InformationalVersion ?? "Unknown";
// Strip commit hash suffix
var mlVersion = mlInfoVer.Contains('+') ? mlInfoVer[..mlInfoVer.IndexOf('+')] : mlInfoVer;
```

### Windows App SDK Version
No runtime API exists to get the friendly version (e.g., "1.8.6"). Options:
- Hardcode it since you control the package reference
- The NuGet version (`1.8.260317003`) is a build number, not human-friendly
- `Microsoft.WindowsAppRuntime.dll` file version (`3.0.0.2602`) uses a different scheme

### .NET, OS, Architecture
```csharp
var dotnet = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription;
var os = Environment.OSVersion.VersionString;
var arch = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString();
```

## Detecting Deployment Mode

```csharp
// Self-contained: core WASDK DLL exists next to the exe
var wasDll = Path.Combine(AppContext.BaseDirectory, "Microsoft.WindowsAppRuntime.dll");
bool isSelfContained = File.Exists(wasDll);
```

## ORT Logging

```csharp
var envOptions = new EnvironmentCreationOptions
{
    logId = "MyApp",
    logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE // Or WARNING, ERROR
};
OrtEnv.CreateInstanceWithOptions(ref envOptions);
```

## Common Error Patterns

| Error | Cause | Fix |
|-------|-------|-----|
| `DllNotFoundException: onnxruntime` | Native DLLs not in output dir | Add `<Content>` items in csproj |
| `System.Numerics.Tensors v9.0.0.0` | Missing NuGet dependency | Add `System.Numerics.Tensors` v10.0.5 |
| `GetEpDevices()` returns 0 | InferenceSession created before OrtEnv init | Call `CreateInstanceWithOptions()` first |
| `[ErrorCode:ShapeInference] Not Implemented` | CompileModel on AUTO EP | Use device-specific EP for compilation |
| XAML crash on startup | XamlControlsResources in unpackaged app | Remove from App.xaml, use empty resources |
