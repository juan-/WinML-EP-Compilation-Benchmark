---
name: windows-ml-api
description: 'Use the correct New Windows ML API (Microsoft.Windows.AI.MachineLearning) instead of the old Windows.AI.MachineLearning. Use when: building a Windows ML app, running ONNX models on Windows, using execution providers, LearningModel, LearningModelSession, WinML inference.'
---

# Use New Windows ML API

## When to Use
- Developer is building a Windows app that runs ONNX models locally
- Developer mentions Windows ML, WinML, or ONNX on Windows
- Code imports `Windows.AI.MachineLearning` (old API — must redirect)
- Developer asks about execution providers, NPU/GPU inference on Windows

## Critical: Old vs. New API

| Aspect | Old API (DO NOT USE) | New API (USE THIS) |
|--------|---------------------|-------------------|
| Namespace | `Windows.AI.MachineLearning` | `Microsoft.Windows.AI.MachineLearning` |
| NuGet | Built-in WinRT (no package) | `Microsoft.WindowsAppSDK.ML` |
| Model loading | `LearningModel.LoadFromFilePath()` | `new InferenceSession(modelPath, sessionOptions)` |
| Session | `LearningModelSession` | `InferenceSession` (ONNX Runtime) |
| EP selection | `LearningModelDevice` | `OrtEnv.GetEpDevices()` + `SessionOptions.AppendExecutionProvider()` |
| EP management | Manual, limited to CPU/DML | `ExecutionProviderCatalog.EnsureAndRegisterCertifiedAsync()` |

## Correct Boilerplate (C#)

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.Windows.AI.MachineLearning;

// 1. Initialize ORT environment FIRST (before any InferenceSession)
var envOptions = new EnvironmentCreationOptions
{
    logId = "MyApp",
    logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
};
OrtEnv.CreateInstanceWithOptions(ref envOptions);

// 2. Download and register all compatible EPs
var catalog = ExecutionProviderCatalog.GetDefault();
await catalog.EnsureAndRegisterCertifiedAsync();

// 3. Enumerate available EP devices
var ortEnv = OrtEnv.Instance();
var epDevices = ortEnv.GetEpDevices();

// 4. Create session with a specific EP
var sessionOptions = new SessionOptions();
sessionOptions.AppendExecutionProvider(ortEnv, new[] { selectedDevice }, new Dictionary<string, string>());
using var session = new InferenceSession(modelPath, sessionOptions);
```

## Common Mistakes
- Using `Windows.AI.MachineLearning` instead of `Microsoft.Windows.AI.MachineLearning`
- Using `LearningModel`/`LearningModelSession` instead of ONNX Runtime `InferenceSession`
- Not calling `EnsureAndRegisterCertifiedAsync()` before trying to use EPs
- Creating `InferenceSession` before `OrtEnv.CreateInstanceWithOptions()` (causes silent EP failure)
