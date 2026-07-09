---
name: windows-ml-ep-selection
description: 'Guide for selecting and understanding Windows ML execution providers. Use when: choosing between CPU GPU NPU, EP selection, OrtEpDevice, GetEpDevices, DirectML, OpenVINO, AUTO vs device-specific EP.'
---

# Windows ML Execution Provider Selection

## When to Use
- Developer needs to choose which EP to use for inference
- Understanding the difference between device-specific and AUTO EPs
- Deciding between DML and OpenVINO for GPU
- Understanding NPU execution providers

## EP Types

### Device-Specific EPs
Target a single hardware device directly. Use these when you know which hardware to target.

- `CPUExecutionProvider (CPU)` — Always available, baseline performance
- `DmlExecutionProvider (GPU)` — DirectML, works on any DirectX 12 GPU
- `OpenVINOExecutionProvider (NPU)` — Intel NPU (Neural Processing Unit)
- `OpenVINOExecutionProvider (GPU)` — Intel integrated/discrete GPU via OpenVINO
- `OpenVINOExecutionProvider (CPU)` — Intel CPU via OpenVINO (may outperform default CPU EP)

### AUTO EPs (Meta-Schedulers)
`OpenVINOExecutionProvider.AUTO (*)` — Dynamically picks the best device at runtime, or splits work across multiple devices.

**Trade-offs of AUTO**:
- ✅ No need to choose hardware — picks the best option automatically
- ❌ Does NOT support model compilation (`CompileModel()` fails)
- ❌ May have higher first-inference latency due to runtime device selection

## How to Enumerate EPs

```csharp
// After EnsureAndRegisterCertifiedAsync()
var ortEnv = OrtEnv.Instance();
var epDevices = ortEnv.GetEpDevices();

foreach (var device in epDevices)
{
    Console.WriteLine($"{device.EpName} ({device.HardwareDevice.Type})");
}
```

## How to Select an EP

```csharp
var sessionOptions = new SessionOptions();
sessionOptions.AppendExecutionProvider(ortEnv, new[] { selectedDevice }, new Dictionary<string, string>());
using var session = new InferenceSession(modelPath, sessionOptions);
```

## Performance Characteristics (ResNet-50, typical x64 machine)

| EP | Inference (Uncompiled) | Inference (Compiled) | Best For |
|----|----------------------|---------------------|----------|
| CPU | ~82 ms | ~67 ms | Reliability, always works |
| DML GPU | ~449 ms (cold) | ~89 ms | GPU-accelerated, broad compatibility |
| OpenVINO NPU | ~12 ms | ~20 ms | Lowest latency, battery efficient |
| OpenVINO GPU | ~2155 ms (cold) | ~73 ms | Best compiled GPU perf on Intel |
| OpenVINO CPU | ~163 ms | ~169 ms | Intel-optimized CPU path |

**Note**: First uncompiled inference on GPU EPs includes JIT compilation overhead. Use compiled mode for GPU EPs.

## Decision Guide

- **Lowest latency**: OpenVINO NPU (if available)
- **Best GPU performance**: Compile + OpenVINO GPU or DML
- **Maximum compatibility**: CPUExecutionProvider (always available)
- **Let the system decide**: OpenVINO.AUTO (but can't use compilation)
- **Broad GPU support (AMD/NVIDIA/Intel)**: DmlExecutionProvider
