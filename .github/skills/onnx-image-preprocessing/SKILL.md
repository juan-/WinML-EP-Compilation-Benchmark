---
name: onnx-image-preprocessing
description: 'Preprocess images for ONNX vision models on Windows using BitmapDecoder and DenseTensor. Use when: image classification, ResNet preprocessing, ImageNet normalization, BGRA to RGB, resize image for ONNX, DenseTensor from image.'
---

# ONNX Image Preprocessing for Windows

## When to Use
- Preparing images for ONNX vision models (ResNet, MobileNet, EfficientNet, etc.)
- Converting Windows `BitmapDecoder` output to ONNX Runtime `DenseTensor<float>`
- Applying ImageNet normalization
- Handling BGRA → RGB channel conversion

## ResNet-50 / ImageNet Preprocessing Pipeline

Most ImageNet-trained models expect:
1. **Resize** to 224x224 pixels
2. **RGB** channel order (Windows images are BGRA)
3. **Normalize** with ImageNet mean/std per channel
4. **CHW layout** (Channels, Height, Width) not HWC

## Complete C# Implementation

```csharp
using Microsoft.ML.OnnxRuntime.Tensors;
using Windows.Graphics.Imaging;
using Windows.Storage;

private static readonly float[] ImageNetMean = { 0.485f, 0.456f, 0.406f };
private static readonly float[] ImageNetStd  = { 0.229f, 0.224f, 0.225f };

public static async Task<DenseTensor<float>> PreprocessImageAsync(string imagePath)
{
    // Load and decode image
    var file = await StorageFile.GetFileFromPathAsync(imagePath);
    using var stream = await file.OpenReadAsync();
    var decoder = await BitmapDecoder.CreateAsync(stream);

    // Resize to 224x224
    var transform = new BitmapTransform
    {
        ScaledWidth = 224,
        ScaledHeight = 224,
        InterpolationMode = BitmapInterpolationMode.Linear
    };
    var pixelData = await decoder.GetPixelDataAsync(
        BitmapPixelFormat.Bgra8,
        BitmapAlphaMode.Premultiplied,
        transform,
        ExifOrientationMode.IgnoreExifOrientation,
        ColorManagementMode.DoNotColorManage);

    var pixels = pixelData.DetachPixelData(); // BGRA byte array

    // Convert to CHW float tensor with ImageNet normalization
    var tensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });

    for (int y = 0; y < 224; y++)
    {
        for (int x = 0; x < 224; x++)
        {
            int i = (y * 224 + x) * 4; // BGRA stride
            float b = pixels[i] / 255f;
            float g = pixels[i + 1] / 255f;
            float r = pixels[i + 2] / 255f;
            // Note: BGRA order — index 0=B, 1=G, 2=R, 3=A

            // ImageNet normalization: (pixel - mean) / std
            tensor[0, 0, y, x] = (r - ImageNetMean[0]) / ImageNetStd[0]; // R channel
            tensor[0, 1, y, x] = (g - ImageNetMean[1]) / ImageNetStd[1]; // G channel
            tensor[0, 2, y, x] = (b - ImageNetMean[2]) / ImageNetStd[2]; // B channel
        }
    }

    return tensor;
}
```

## Common Mistakes
- **BGRA vs RGB**: Windows `BitmapDecoder` outputs BGRA. Channel 0 is Blue, not Red.
- **Missing normalization**: Raw 0-255 values without ImageNet mean/std subtraction will produce wrong predictions.
- **HWC vs CHW**: ONNX models typically expect CHW layout `[batch, channels, height, width]`, not HWC.
- **Wrong input size**: Check the model's input metadata — ResNet-50 expects 224x224, but other models may differ.

## Output Processing (Softmax + Top-K)

```csharp
// Get raw logits from model output
var output = results.First().AsEnumerable<float>().ToArray();

// Apply softmax
float maxLogit = output.Max();
var expScores = output.Select(x => MathF.Exp(x - maxLogit)).ToArray();
float sumExp = expScores.Sum();
var probabilities = expScores.Select(e => e / sumExp).ToArray();

// Get top-5 predictions
var top5 = probabilities
    .Select((prob, idx) => (prob, idx))
    .OrderByDescending(x => x.prob)
    .Take(5);
```
