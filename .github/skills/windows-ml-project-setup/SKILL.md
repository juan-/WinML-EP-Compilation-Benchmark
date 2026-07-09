---
name: windows-ml-project-setup
description: 'Set up a .NET/WinUI 3 project for Windows ML with correct NuGet packages, csproj settings, and native DLL deployment. Use when: creating a new Windows ML project, fixing build errors, DllNotFoundException for onnxruntime, self-contained deployment, unpackaged WinUI 3 app setup.'
---

# Windows ML Project Setup

## When to Use
- Creating a new C# project that uses Windows ML / ONNX Runtime
- Getting build errors related to PRI generation, MSIX, or MRT
- Getting `DllNotFoundException` for `onnxruntime` at runtime
- Getting `FileNotFoundException` for `System.Numerics.Tensors`
- Setting up self-contained or framework-dependent deployment

## Required NuGet Packages

```xml
<PackageReference Include="Microsoft.WindowsAppSDK" Version="1.8.260317003" />
<PackageReference Include="Microsoft.WindowsAppSDK.ML" Version="1.8.2141" />
<PackageReference Include="Microsoft.Windows.SDK.BuildTools" Version="10.0.26100.4654" />
<PackageReference Include="System.Numerics.Tensors" Version="10.0.5" />
```

**Important**: `System.Numerics.Tensors` is NOT declared as a transitive dependency by the ML package. You must add it explicitly or you'll get a runtime crash: `Could not load file or assembly 'System.Numerics.Tensors, Version=9.0.0.0'`.

## Required csproj Settings for Unpackaged Apps

```xml
<PropertyGroup>
    <OutputType>WinExe</OutputType>
    <TargetFramework>net8.0-windows10.0.22621.0</TargetFramework>
    <UseWinUI>true</UseWinUI>
    <WindowsAppSDKSelfContained>true</WindowsAppSDKSelfContained>
    <WindowsPackageType>None</WindowsPackageType>
    <EnableMsixTooling>false</EnableMsixTooling>
    <GeneratePriFile>false</GeneratePriFile>
    <DisableMrtCodeGen>true</DisableMrtCodeGen>
    <EnableCoreMrtTooling>false</EnableCoreMrtTooling>
</PropertyGroup>
```

## Native DLL Deployment (Critical)

The `Microsoft.WindowsAppSDK.ML` NuGet package stores native DLLs in a non-standard path (`runtimes-framework\win-x64\native\`) that .NET doesn't auto-copy. You MUST add these manually:

```xml
<ItemGroup>
    <Content Include="$(NuGetPackageRoot)microsoft.windowsappsdk.ml\1.8.2141\runtimes-framework\win-x64\native\onnxruntime.dll"
             CopyToOutputDirectory="PreserveNewest" Link="onnxruntime.dll" />
    <Content Include="$(NuGetPackageRoot)microsoft.windowsappsdk.ml\1.8.2141\runtimes-framework\win-x64\native\onnxruntime_providers_shared.dll"
             CopyToOutputDirectory="PreserveNewest" Link="onnxruntime_providers_shared.dll" />
    <Content Include="$(NuGetPackageRoot)microsoft.windowsappsdk.ml\1.8.2141\runtimes-framework\win-x64\native\DirectML.dll"
             CopyToOutputDirectory="PreserveNewest" Link="DirectML.dll" />
    <Content Include="$(NuGetPackageRoot)microsoft.windowsappsdk.ml\1.8.2141\runtimes-framework\win-x64\native\Microsoft.Windows.AI.MachineLearning.dll"
             CopyToOutputDirectory="PreserveNewest" Link="Microsoft.Windows.AI.MachineLearning.dll" />
</ItemGroup>
```

For ARM64, replace `win-x64` with `win-arm64`.

## XAML Theme Warning

In unpackaged apps without the full VS toolchain, `XamlControlsResources` in `App.xaml` can cause crashes. Use empty resources:

```xml
<Application.Resources>
    <ResourceDictionary />
</Application.Resources>
```

## Build Command

```
dotnet build MyProject.csproj -c Debug -p:Platform=x64
```
