using Microsoft.Windows.AI.MachineLearning;

namespace WinMLResNet;

/// <summary>
/// Information about one execution provider discovered in the Windows ML EP catalog.
/// </summary>
public class EpRegistrationInfo
{
    public string Name { get; set; } = "";
    public string PackageFamily { get; set; } = "";
    public string LibraryPath { get; set; } = "";
    public bool Registered { get; set; }
    public string? Error { get; set; }

    public string Summary =>
        $"{Name}: {(Registered ? "registered" : "not registered")}" +
        (string.IsNullOrEmpty(Error) ? "" : $" ({Error})");
}

/// <summary>
/// Registers installed Windows ML execution providers with ONNX Runtime using the official
/// <see cref="ExecutionProviderCatalog"/> flow — enumerate with
/// <see cref="ExecutionProviderCatalog.FindAllProviders"/>, make each ready with
/// <see cref="ExecutionProvider.EnsureReadyAsync"/>, then register with
/// <see cref="ExecutionProvider.TryRegister"/>. This is the pattern documented for
/// <c>ExecutionProvider.TryRegister</c>:
/// https://learn.microsoft.com/windows/windows-app-sdk/api/winrt/microsoft.windows.ai.machinelearning.executionprovider.tryregister
///
/// <para><c>TryRegister()</c> returns <c>true</c> on success and <c>false</c> if the provider
/// couldn't be registered (for example a vendor EP whose hardware/driver isn't present), so every
/// installed provider is surfaced without throwing. Providers that register are then enumerable via
/// <c>OrtEnv.GetEpDevices()</c>. The built-in CPU and DirectML providers are always available with
/// no registration.</para>
/// </summary>
public static class EpRegistration
{
    // Registration is process-global in ORT, so guard against re-registering the same provider
    // (both ImageClassifier and TextEmbedder initialize independently).
    private static readonly HashSet<string> s_registered = new(StringComparer.OrdinalIgnoreCase);
    private static readonly object s_lock = new();

    /// <summary>Results of the most recent full <see cref="RegisterAllAsync"/> pass, for UI diagnostics.</summary>
    public static IReadOnlyList<EpRegistrationInfo> LastResults { get; private set; } = Array.Empty<EpRegistrationInfo>();

    /// <summary>Registers every installed execution provider (catalog <c>FindAllProviders</c> +
    /// <c>EnsureReadyAsync</c> + <c>TryRegister</c>).</summary>
    public static Task<List<EpRegistrationInfo>> RegisterAllAsync() => RegisterAsync(null);

    /// <summary>
    /// Registers only the named providers. Used by the app to register the known-safe set
    /// (providers a crash-isolated worker confirmed produce a device), so the UI process never
    /// tries to load a fragile vendor library that could native-crash on hardware without the
    /// matching accelerator.
    /// </summary>
    public static async Task RegisterSpecificAsync(IEnumerable<string> providerNames)
    {
        var set = new HashSet<string>(providerNames, StringComparer.OrdinalIgnoreCase);
        await RegisterAsync(set);
    }

    private static async Task<List<EpRegistrationInfo>> RegisterAsync(HashSet<string>? onlyProviders)
    {
        var results = new List<EpRegistrationInfo>();

        ExecutionProviderCatalog catalog;
        try
        {
            catalog = ExecutionProviderCatalog.GetDefault();
        }
        catch (Exception ex)
        {
            var fail = new List<EpRegistrationInfo> { new() { Name = "(catalog)", Error = ex.Message } };
            if (onlyProviders is null) LastResults = fail;
            return fail;
        }

        ExecutionProvider[] providers;
        try
        {
            providers = catalog.FindAllProviders();
        }
        catch (Exception ex)
        {
            var fail = new List<EpRegistrationInfo> { new() { Name = "(FindAllProviders)", Error = ex.Message } };
            if (onlyProviders is null) LastResults = fail;
            return fail;
        }

        foreach (var provider in providers)
        {
            if (onlyProviders is not null && !onlyProviders.Contains(provider.Name)) continue;

            var info = new EpRegistrationInfo { Name = provider.Name };
            try { info.PackageFamily = provider.PackageId?.FamilyName ?? ""; } catch { /* not ready yet */ }

            // Already registered earlier in this process — TryRegister is process-global.
            lock (s_lock)
            {
                if (s_registered.Contains(provider.Name))
                {
                    info.Registered = true;
                    results.Add(info);
                    continue;
                }
            }

            try
            {
                // Make the provider ready (installs/stages the library if needed). We only advance
                // to TryRegister for providers already present on the system; acquisition of a
                // NotPresent provider is intentionally left to the catalog's certified flow so this
                // never blocks on a download.
                if (provider.ReadyState == ExecutionProviderReadyState.NotReady)
                {
                    var ready = await provider.EnsureReadyAsync();
                    if (ready.Status == ExecutionProviderReadyResultState.Failure)
                        info.Error = $"EnsureReady failed (0x{ready.ExtendedError:X8})";
                }

                if (provider.ReadyState == ExecutionProviderReadyState.Ready)
                {
                    bool registered = provider.TryRegister();
                    info.Registered = registered;
                    if (registered)
                    {
                        lock (s_lock) { s_registered.Add(provider.Name); }
                    }
                    else if (string.IsNullOrEmpty(info.Error))
                    {
                        info.Error = "TryRegister returned false (provider unavailable on this system)";
                    }
                }
                else if (string.IsNullOrEmpty(info.Error))
                {
                    info.Error = $"not ready ({provider.ReadyState})";
                }
            }
            catch (Exception ex)
            {
                info.Error = ex.Message.Replace("\r", " ").Replace("\n", " ");
            }

            results.Add(info);
        }

        if (onlyProviders is null) LastResults = results;
        return results;
    }
}
