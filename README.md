# WinML EP Benchmarking Sample App

A WinUI 3 desktop sample that benchmarks ONNX model inference across **CPU, GPU, and NPU** execution providers using the new **Windows ML API** (`Microsoft.Windows.AI.MachineLearning`).

The app ships two end-to-end scenarios that share the same EP-benchmark UX:

| Scenario | Default Model | Architecture | Input |
|---|---|---|---|
| **Image Classification** | ResNet-50 v2 | Convolutional Neural Network (~25M params) | JPG / PNG / BMP image |
| **Text Similarity** | sentence-transformers/all-MiniLM-L6-v2 | Distilled BERT transformer (~22M params, 384-dim embeddings) | Free-form text query |

Both tabs let you compare every detected execution provider side-by-side in JIT and AOT compilation modes, with sortable columns, an emoji heat map, and CSV export. A third **EP Compilation** tab focuses purely on ahead-of-time *compile time* across any set of models and execution providers.

## Prerequisites

- Windows 11 (or Windows 10 22H2+)
- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)
- Visual Studio 2022 with the *.NET desktop development* and *Windows App SDK C# Templates* workloads, **or** VS Code with the C# Dev Kit

## Build & Run

```powershell
dotnet restore
dotnet build WinMLResNet.csproj -c Debug /p:Platform=x64
# Also build the isolated compile worker used by the EP Compilation tab:
dotnet build CompileErrorTest\CompileErrorTest.csproj -c Debug /p:Platform=x64
.\bin\x64\Debug\net8.0-windows10.0.22621.0\AT-WinML-Benchmark-Sample.exe
```

Or open `WinML Bug Bash.sln` in Visual Studio and press F5.

> The output exe is `AT-WinML-Benchmark-Sample.exe` (set via `<AssemblyName>` in the csproj). The csproj filename and solution name are unchanged for git history reasons.

## Models

### Image Classification — bundled

`Assets/resnet50-v2-7.onnx` is committed to the repo (~98 MB). No download required. Use the **Change Model…** button on the Image Classification tab to swap in any other ImageNet-compatible ONNX model.

### Text Similarity — auto-downloads on first use

The first time you visit the **Text Similarity** tab, the app fetches `all-MiniLM-L6-v2.onnx` and `vocab.txt` from HuggingFace into `Assets/` (~90 MB total) and loads them. Subsequent launches reuse the cached files.

The **Coffee Shop Knowledge Bank** corpus (`Assets/embeddings_corpus.txt`) is shipped with the app — ~190 sample customer utterances covering drinks, customizations, food items, payment, and store-info questions. Use the **Download Corpus** button on the tab to save a copy locally and edit it.

## Execution Providers

Windows ML acquires execution providers as separate MSIX packages. Rather than going through the `ExecutionProviderCatalog` WinRT helper, this app registers **every installed EP directly with ONNX Runtime** (see [`EpRegistration.cs`](EpRegistration.cs)):

1. Enumerate installed packages with `PackageManager.FindPackagesForUser(...)`.
2. For each Windows ML EP package, read its `AppxManifest.xml` from [`Package.InstalledLocation`](https://learn.microsoft.com/uwp/api/windows.applicationmodel.package.installedlocation) and pull the provider registration name and DLL from the `com.microsoft.windowsmlruntime.executionprovider[.N]` package extension.
3. Build `<InstalledLocation>\ExecutionProvider\<provider>.dll` and register it via `OrtEnv.RegisterExecutionProviderLibrary(name, path)`.

This registers the provider library regardless of certification state or whether matching hardware is present, so every installed EP is surfaced through `OrtEnv.GetEpDevices()`. A vendor EP whose native dependencies or driver aren't present fails to load with a **catchable** exception (e.g. *"…depends on cudart64_12.dll which is missing"*) — it's skipped, never crashing the app. The built-in **CPU** (`CPUExecutionProvider`) and **DirectML** (`DmlExecutionProvider`) EPs are always available with no install or registration.

To add the vendor EPs (Intel OpenVINO, Qualcomm QNN, NVIDIA TensorRT-RTX, AMD MIGraphX / VitisAI, WinML CG), sideload their MSIX packages:

```powershell
Add-AppxPackage -Path <path-to>\OpenVINOEP.<ver>.msix
# …repeat for each EP package…
```

> **Which EPs appear depends on your hardware.** A provider *registers* only if its native library (and its dependencies) load; it becomes a *selectable device* only if it also enumerates compatible hardware. On a machine without a given vendor's GPU/NPU, that EP won't produce a benchmarkable device even after its package is installed — this is expected.

## What the App Shows

### Image Classification tab
- Pick an image, choose an EP, and see the top-5 ImageNet predictions with confidence scores.
- **Compare All EPs** runs every detected execution provider in both JIT (implicit, compile-on-load) and AOT (explicit `OrtModelCompilationOptions.CompileModel()`) modes, cold and warm.

### Text Similarity tab
- Type any phrase (e.g. *"matcha latte with soy milk"*).
- The model embeds it and ranks the **top-5 most semantically similar** entries from the coffee-shop corpus by cosine similarity.
- **Compare All EPs** benchmarks every EP exactly like the classification tab.

### EP Compilation tab
- Measures how long each execution provider takes to **ahead-of-time compile** a model via `OrtModelCompilationOptions.CompileModel()` — the key first-run cost for AOT deployment — for **different models**.
- Choose one or more models: the bundled ResNet-50 and MiniLM, any models found under `Assets/certification-models/` (auto-discovered on startup), your own `.onnx` files (**Add .onnx File…**), or an entire folder (**Add Folder…**).
- Tick which EPs to test (CPU is selected by default) and click **Run Compilation Benchmark** to build a Model × EP compile-time table, with compiled-artifact sizes and CSV export.
- Every compile runs in an **isolated worker process** (`CompileErrorTest.exe`). If a provider native-crashes — common for a GPU/NPU EP on a machine with no matching hardware — it's reported as `Crashed` in the table instead of taking down the app.

### CPU compilation: built-in MLAS vs. vendor CPU (e.g. OpenVINO)

There are **two distinct CPU compile paths**, and the tab exposes both:

- **`CPUExecutionProvider (CPU)`** — ONNX Runtime's built-in **MLAS** CPU provider. It's statically linked into `onnxruntime.dll`, so it's always present; a default `SessionOptions` already includes it with no registration.
- **`OpenVINOExecutionProvider (CPU)`** — OpenVINO's **own** CPU runtime (oneDNN-based), a genuinely different compile path from MLAS. It's selected exactly the way Windows ML's own tooling does it: find the `OrtEpDevice` whose `EpName == "OpenVINOExecutionProvider"` **and** `HardwareDevice.Type == CPU`, then `AppendExecutionProvider(env, [device], options)` — never the built-in CPU EP.

A vendor CPU device only becomes selectable when the provider **enumerates it** via `GetEpDevices()`, which requires matching hardware (OpenVINO gates on Intel). The tab still lists `OpenVINO (CPU/GPU/NPU)` and other registered vendor targets so you can select them explicitly; on hardware that enumerates the device you get real numbers, and on hardware that doesn't the row reports *"OpenVINOExecutionProvider did not enumerate a CPU device on this system"* rather than silently falling back to MLAS. (Microsoft's own `winml compile --ep openvino --device cpu` behaves identically — it reports the EP as unavailable when no device enumerates.)

### Certification models (Evergreen + samples)

The WinML certification spec's **Evergreen (EG)** set is 12 foundational models; the 7 vision/encoder ones (ViT, ResNet-50, 2× BERT, 3× CLIP) are the models that get a *compilation* evaluation. They're published on HuggingFace as PyTorch source, so they must be exported to ONNX first with the [`winml` CLI](https://pypi.org/project/winml-cli/):

```powershell
# One-time toolchain setup
irm https://astral.sh/uv/install.ps1 | iex
uv venv --python 3.11
uv pip install winml-cli

# Export a model to ONNX (repeat per model id)
winml export -m google/vit-base-patch16-224 -o Assets/certification-models/eg-vit-base-patch16-224/model.onnx
```

Exported models land under `Assets/certification-models/` and are picked up automatically the next time the app starts. The two bundled SqueezeNet samples (`sample-squeezenet`, `sample-squeezenet-fp32`) come from the WinML model catalog and need no export. These large models are intentionally **excluded from the build copy** and discovered from the source `Assets/` folder at runtime.

The 5 LLM Evergreen models (Phi, Llama, Qwen, DeepSeek) and the **ASG production models** (PSAPI / Windows AI API) are out of scope for this compile benchmark: LLMs run through ORT-GenAI rather than `CompileModel()`, and ASG models are internal NPU builds distributed via PSAPI. To benchmark an ASG drop when you have one, point **Add Folder…** at its directory.

### Benchmark table columns

| Column | Meaning |
|---|---|
| `#` | Rank in the current sort order |
| Execution Provider | EP used (CPU, DML, OpenVINO, QNN, …) |
| Mode | One of `JIT (Cold)`, `AOT (Cold)`, `Warm (JIT-built)`, `Warm (AOT-built)`. Cold rows do the actual compile/load work; warm rows reuse the cached `InferenceSession`. |
| Image Preprocessing / Tokenization | Pre-inference data prep (CPU-only) |
| Session Creation | First-time session init or cache hit. On `JIT (Cold)` for vendor EPs, the EP's compile cost is folded into this column. |
| AOT Compile | Time spent in `OrtModelCompilationOptions.CompileModel()`. Shows `XXX ms` on `AOT (Cold)`, `AOT cached` on `Warm (AOT-built)`, `during session creation` on `JIT (Cold)`, `session cached` on `Warm (JIT-built)`. |
| Inference | Core EP-dependent metric — **lower is better** |
| EP Total | Session Creation + AOT Compile + Inference (default sort). The total time the EP itself spends on the model, excluding app-level overhead. On warm rows this collapses to just Inference. |
| Total | End-to-end wall-clock time including preprocessing and post-processing |
| Memory Δ | Working-set memory change (MB) before/after the run — captures both managed and native allocations. An asterisk (`*`) marks the first (cold) run for that EP+Mode, where the model is loaded into memory; subsequent runs reuse the cached session and show near-zero deltas. |
| Top Prediction / Top Match | Best label or corpus sentence |
| Confidence / Similarity | Score for the top result |

### Cold vs warm runs

Each EP is benchmarked in four rows:

- **`JIT (Cold)`** — first run, no AOT artifact; the EP compiles the graph implicitly inside `new InferenceSession(...)`. Compile cost shows up under Session Creation.
- **`AOT (Cold)`** — first run, model precompiled via `OrtModelCompilationOptions.CompileModel()` into a `*_ctx.onnx` artifact, which is then loaded into a session. Compile cost shows up under AOT Compile.
- **`Warm (JIT-built)` / `Warm (AOT-built)`** — the cached `InferenceSession` from the matching cold run is reused; no compilation or session construction. The `-built` suffix identifies which lineage produced the cached session, not work happening at runtime.

Cold rows show the first-time-use cost a real user would pay; warm rows show steady-state inference performance and are the better number for comparing EPs.

**Compare All EPs** clears the session cache before running so every EP gets an honest cold/warm pair.

The Glossary tab inside the app explains everything in detail.

## Project Structure

| File | Purpose |
|---|---|
| `App.xaml` / `App.xaml.cs` | Application entry point |
| `MainWindow.xaml` / `MainWindow.xaml.cs` | All pivot tabs and event handlers |
| `ImageClassifier.cs` | ResNet inference + per-EP benchmark harness |
| `TextEmbedder.cs` | MiniLM tokenize → embed → cosine search + auto-download |
| `EpRegistration.cs` | Force-registers every installed execution provider |
| `CompileBenchmark.cs` | EP Compilation tab data model + isolated-worker runner |
| `CompileErrorTest/` | Headless worker: compiles a model on one EP in an isolated process (used by the EP Compilation tab); also doubles as a standalone EP enumeration/compile probe |
| `Assets/resnet50-v2-7.onnx` | Bundled image classification model |
| `Assets/imagenet_labels.txt` | 1,000 ImageNet class labels |
| `Assets/embeddings_corpus.txt` | Coffee-shop knowledge bank |
| `Assets/app-icon.ico` | App icon |
| `WinMLResNet.csproj` | Project file (output exe: `AT-WinML-Benchmark-Sample.exe`) |

## Key NuGet Packages

| Package | Purpose |
|---|---|
| `Microsoft.WindowsAppSDK` 2.0.1 | WinUI 3 desktop runtime |
| `Microsoft.WindowsAppSDK.ML` 2.0.300 | Windows ML projection (`Microsoft.Windows.AI.MachineLearning`) |
| `Microsoft.ML.Tokenizers` 1.0.2 | BERT WordPiece tokenizer for text embeddings |
| `System.Numerics.Tensors` 10.0.5 | Tensor helpers |

## Copilot Agent Skills

This repo includes 7 GitHub Copilot agent skills in `.github/skills/` distilled from lessons learned building this app. They help AI coding agents avoid common pitfalls when developing Windows ML applications:

- **windows-ml-api** — Use the new Windows ML API, not the legacy WinRT one
- **windows-ml-project-setup** — NuGet packages, csproj settings, native DLL deployment
- **ort-init-order** — Initialize the ORT environment before creating any `InferenceSession`
- **onnx-model-compilation** — Which EPs support compilation and best practices
- **windows-ml-ep-selection** — Choosing between CPU, GPU, NPU, and AUTO EPs
- **onnx-image-preprocessing** — Image-to-tensor pipeline for vision models
- **windows-ml-debugging** — Version detection, deployment mode, common errors

## License

This sample application is released under the [MIT License](LICENSE). See the **About** tab inside the app for full third-party model and library attributions (ResNet-50, MiniLM, ONNX Runtime, Windows App SDK, Microsoft.ML.Tokenizers, etc.).

Learn more about Windows ML at [aka.ms/winml](https://aka.ms/winml).
