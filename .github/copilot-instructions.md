# WinML ResNet Image Classifier

## Project Overview
- **Type**: Windows Desktop App (C# / WinUI 3 with Windows App SDK)
- **Purpose**: Image classification using ResNet50 ONNX model via Windows ML
- **Framework**: .NET 8 + Windows App SDK + Microsoft.AI.MachineLearning

## Build & Run
- Build: `dotnet build`
- Run: `dotnet run` or launch via Visual Studio / VS Code debugger

## Key Conventions
- ONNX model goes in the `Assets/` folder
- Image classification results show top-5 predictions with confidence scores
- Use `LearningModelSession` for WinML inference
