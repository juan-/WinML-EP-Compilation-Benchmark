---
name: ort-init-order
description: 'Correct ONNX Runtime environment initialization order to avoid silent EP registration failure. Use when: GetEpDevices returns 0 devices, execution providers not found, OrtEnv initialization, InferenceSession before CreateInstanceWithOptions.'
---

# ORT Environment Initialization Order

## When to Use
- `OrtEnv.Instance().GetEpDevices()` returns 0 devices after calling `EnsureAndRegisterCertifiedAsync()`
- Execution providers aren't showing up despite successful registration
- Any code that creates `InferenceSession` for metadata reading before EP setup

## The Rule

**`OrtEnv.CreateInstanceWithOptions()` MUST be called before ANY `InferenceSession` is created.**

Creating an `InferenceSession` (even for reading model metadata) implicitly initializes the ORT environment with default options. Once initialized, `CreateInstanceWithOptions()` silently becomes a no-op. EPs registered after this point won't appear in `GetEpDevices()`.

## Correct Order

```csharp
// ✅ CORRECT: Init ORT environment first
var envOptions = new EnvironmentCreationOptions
{
    logId = "MyApp",
    logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
};
OrtEnv.CreateInstanceWithOptions(ref envOptions);  // MUST be first

// Now safe to read model metadata
using var metadataSession = new InferenceSession(modelPath);
var meta = metadataSession.ModelMetadata;

// Now register EPs
var catalog = ExecutionProviderCatalog.GetDefault();
await catalog.EnsureAndRegisterCertifiedAsync();

// EPs will be available
var devices = OrtEnv.Instance().GetEpDevices();  // Returns N devices
```

## Wrong Order (Silent Failure)

```csharp
// ❌ WRONG: InferenceSession created first
using var metadataSession = new InferenceSession(modelPath);  // Implicitly inits ORT env
var meta = metadataSession.ModelMetadata;

// This is now a no-op!
OrtEnv.CreateInstanceWithOptions(ref envOptions);

await catalog.EnsureAndRegisterCertifiedAsync();
var devices = OrtEnv.Instance().GetEpDevices();  // Returns 0 devices!
```

## Best Practice

Put `OrtEnv.CreateInstanceWithOptions()` in your classifier/service constructor, before any other ORT calls:

```csharp
public class MyClassifier
{
    public MyClassifier()
    {
        var envOptions = new EnvironmentCreationOptions { logId = "MyApp" };
        OrtEnv.CreateInstanceWithOptions(ref envOptions);
    }
}
```

## Symptoms of This Bug
- `GetEpDevices()` returns an empty list
- No error or exception is thrown anywhere
- `EnsureAndRegisterCertifiedAsync()` completes successfully
- Everything appears normal except no EPs are available
