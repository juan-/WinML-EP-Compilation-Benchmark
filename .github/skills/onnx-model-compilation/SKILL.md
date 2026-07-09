---
name: onnx-model-compilation
description: 'Best practices for ONNX model compilation with OrtModelCompilationOptions. Use when: compiling ONNX models, OrtModelCompilationOptions, CompileModel, model optimization, EP compilation support, ShapeInference error.'
---

# ONNX Model Compilation Best Practices

## When to Use
- Developer wants to optimize model performance with `OrtModelCompilationOptions`
- Getting `[ErrorCode:ShapeInference] Not Implemented` errors during compilation
- Deciding whether to use compiled vs. uncompiled inference
- Benchmarking or comparing EP performance

## EP Compilation Support

Not all EPs support `OrtModelCompilationOptions.CompileModel()`. As of WASDK 1.8.6:

| EP | Compilation | Notes |
|----|------------|-------|
| CPUExecutionProvider | ✅ Supported | ~850 ms typical |
| DmlExecutionProvider (GPU) | ✅ Supported | ~980 ms typical |
| OpenVINOExecutionProvider (NPU) | ✅ Supported | ~6400 ms typical |
| OpenVINOExecutionProvider (GPU) | ✅ Supported | ~915 ms typical |
| OpenVINOExecutionProvider (CPU) | ✅ Supported | ~730 ms typical |
| OpenVINOExecutionProvider.AUTO (*) | ❌ Fails | "Not Implemented" error |

**The `.AUTO` variants do not support compilation** because AUTO is a meta-scheduler that defers device selection to runtime — you can't pre-compile for a device when the purpose is to choose the device later.

## Always Wrap in Try/Catch

Since there's no `SupportsCompilation` property on `OrtEpDevice`, always use try/catch:

```csharp
try
{
    using var compileOptions = new OrtModelCompilationOptions(sessionOptions);
    compileOptions.SetInputModelPath(modelPath);
    compileOptions.SetOutputModelPath(compiledPath);
    compileOptions.CompileModel();
}
catch (Exception ex)
{
    // Fall back to uncompiled inference
    // Log: this EP doesn't support compilation
}
```

## Why Compilation Matters (GPU EPs)

GPU execution providers have significant cold-start penalties without compilation:

| EP | Uncompiled | Compiled | Speedup |
|----|-----------|----------|---------|
| DML GPU | ~449 ms | ~89 ms | **5x** |
| OpenVINO GPU | ~2155 ms | ~73 ms | **29x** |

For GPU EPs, always prefer compiled mode.

## Compilation Pattern

```csharp
var compiledPath = Path.Combine(cacheDir, $"{modelName}-compiled-{epName}.onnx");

// Compile once, cache to disk
using var compileOptions = new OrtModelCompilationOptions(sessionOptions);
compileOptions.SetInputModelPath(originalModelPath);
compileOptions.SetOutputModelPath(compiledPath);
compileOptions.CompileModel();

// Load compiled model for inference
using var session = new InferenceSession(compiledPath, sessionOptions);
```

## Tips
- Compilation can take seconds to minutes — do it on a background thread
- Cache compiled models to disk and reuse across app launches
- EP or runtime updates may require recompilation
- Compile per-EP — a model compiled for DML won't work optimally on OpenVINO
