using System.Xml.Linq;
using Microsoft.ML.OnnxRuntime;
using Windows.Management.Deployment;

namespace WinMLResNet;

/// <summary>
/// Information about one execution provider discovered from an installed MSIX package.
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
/// Registers every installed Windows ML execution provider directly with ONNX Runtime, by
/// resolving each EP's MSIX package install location and pointing ORT at the provider DLL.
///
/// <para>How it works:</para>
/// <list type="number">
///   <item>Enumerate installed packages with <see cref="PackageManager.FindPackagesForUser(string)"/>.</item>
///   <item>For each Windows ML EP package, read its <c>AppxManifest.xml</c> from
///     <c>Package.InstalledLocation</c> and pull the provider registration name, the public folder,
///     and the provider DLL name from the
///     <c>com.microsoft.windowsmlruntime.executionprovider[.N]</c> package extension.</item>
///   <item>Build <c>&lt;InstalledLocation&gt;\&lt;PublicFolder&gt;\&lt;ProviderPath&gt;</c> and register it via
///     <see cref="OrtEnv.RegisterExecutionProviderLibrary(string, string)"/>.</item>
/// </list>
///
/// <para>This is a deliberately lower-level alternative to
/// <c>ExecutionProviderCatalog.EnsureAndRegisterCertifiedAsync()</c>: it registers the provider
/// library regardless of certification state or whether matching hardware is present, so every
/// installed EP is available for enumeration via <see cref="OrtEnv.GetEpDevices"/>. The built-in
/// CPU and DirectML providers are always present without any registration.</para>
/// </summary>
public static class EpRegistration
{
    // Windows ML EP packages declare this package extension. v1 packages use the bare name;
    // versioned packages append a suffix (e.g. ".2"), so we match by prefix.
    private const string EpExtensionPrefix = "com.microsoft.windowsmlruntime.executionprovider";

    // Registration is process-global in ORT, so guard against re-registering the same provider
    // (both ImageClassifier and TextEmbedder initialize independently).
    private static readonly HashSet<string> s_registered = new(StringComparer.OrdinalIgnoreCase);
    private static readonly object s_lock = new();

    /// <summary>Results of the most recent <see cref="RegisterAll"/> pass, for UI diagnostics.</summary>
    public static IReadOnlyList<EpRegistrationInfo> LastResults { get; private set; } = Array.Empty<EpRegistrationInfo>();

    public static Task<List<EpRegistrationInfo>> RegisterAllAsync() => Task.Run(RegisterAll);

    public static List<EpRegistrationInfo> RegisterAll() => Register(null);

    /// <summary>
    /// Registers only the named providers in-process. Used by the app to register the known-safe
    /// set (providers that a crash-isolated worker confirmed produce a device), so the UI process
    /// never force-loads a fragile vendor library that could native-crash on hardware without the
    /// matching accelerator.
    /// </summary>
    public static Task RegisterSpecificAsync(IEnumerable<string> providerNames)
    {
        var set = new HashSet<string>(providerNames, StringComparer.OrdinalIgnoreCase);
        return Task.Run(() => Register(set));
    }

    private static List<EpRegistrationInfo> Register(HashSet<string>? onlyProviders)
    {
        var ortEnv = OrtEnv.Instance();

        IReadOnlyList<Windows.ApplicationModel.Package> packages;
        try
        {
            var pm = new PackageManager();
            packages = pm.FindPackagesForUser(string.Empty).ToList();
        }
        catch (Exception ex)
        {
            var fail = new List<EpRegistrationInfo> { new() { Name = "(package enumeration)", Error = ex.Message } };
            if (onlyProviders is null) LastResults = fail;
            return fail;
        }

        // Phase 1: collect every (provider, library) candidate declared across all EP packages.
        // A provider can be declared under both the v1 and v2 extension (sometimes pointing at
        // different DLLs), and different packages can offer the same provider — so gather them all
        // and group by provider name.
        var candidates = new Dictionary<string, List<(string family, string libraryPath)>>(StringComparer.OrdinalIgnoreCase);
        foreach (var pkg in packages)
        {
            string family;
            try { family = pkg.Id.FamilyName; } catch { continue; }

            // Cheap pre-filter so we don't read every manifest on the machine. The manifest parse
            // below is the real authority — non-EP packages simply expose no EP extension.
            if (!LooksLikeEpPackage(family)) continue;

            string installPath;
            try { installPath = pkg.InstalledLocation.Path; }
            catch { continue; }

            var manifestPath = Path.Combine(installPath, "AppxManifest.xml");
            if (!File.Exists(manifestPath)) continue;

            foreach (var (providerName, publicFolder, providerPath) in ParseEpExtensions(manifestPath))
            {
                if (onlyProviders is not null && !onlyProviders.Contains(providerName)) continue;

                var libraryPath = Path.Combine(installPath, publicFolder, providerPath);
                if (!candidates.TryGetValue(providerName, out var list))
                    candidates[providerName] = list = new List<(string, string)>();
                if (!list.Any(c => string.Equals(c.libraryPath, libraryPath, StringComparison.OrdinalIgnoreCase)))
                    list.Add((family, libraryPath));
            }
        }

        // Phase 2: register one library per provider. Try each candidate DLL until one loads, so a
        // provider declared under multiple extensions still registers if any variant is usable.
        var results = new List<EpRegistrationInfo>();
        foreach (var (providerName, list) in candidates)
        {
            var info = new EpRegistrationInfo { Name = providerName, PackageFamily = list[0].family, LibraryPath = list[0].libraryPath };

            lock (s_lock)
            {
                if (s_registered.Contains(providerName))
                {
                    info.Registered = true;
                    results.Add(info);
                    continue;
                }

                string? lastError = null;
                foreach (var (family, libraryPath) in list)
                {
                    try
                    {
                        if (!File.Exists(libraryPath)) { lastError = "provider library not found"; continue; }
                        ortEnv.RegisterExecutionProviderLibrary(providerName, libraryPath);
                        info.Registered = true;
                        info.PackageFamily = family;
                        info.LibraryPath = libraryPath;
                        s_registered.Add(providerName);
                        break;
                    }
                    catch (Exception ex)
                    {
                        // A vendor EP whose native dependencies/driver aren't present can fail to
                        // load here; that's expected on hardware without the matching accelerator.
                        lastError = ex.Message.Replace("\r", " ").Replace("\n", " ");
                    }
                }

                if (!info.Registered) info.Error = lastError;
            }

            results.Add(info);
        }

        if (onlyProviders is null) LastResults = results;
        return results;
    }

    private static bool LooksLikeEpPackage(string family) =>
        family.Contains("WinML", StringComparison.OrdinalIgnoreCase) ||
        family.Contains(".EP.", StringComparison.OrdinalIgnoreCase) ||
        family.Contains("ExecutionProvider", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Reads the <c>com.microsoft.windowsmlruntime.executionprovider[.N]</c> package extensions from
    /// an AppxManifest and yields (providerName, publicFolder, providerPath) for each.
    /// </summary>
    private static IEnumerable<(string ProviderName, string PublicFolder, string ProviderPath)> ParseEpExtensions(string manifestPath)
    {
        XDocument doc;
        try { doc = XDocument.Load(manifestPath); }
        catch { yield break; }

        // Namespace-agnostic: match by local name so we don't depend on the uapNN prefix/version.
        foreach (var ext in doc.Descendants().Where(e => e.Name.LocalName == "PackageExtension"))
        {
            var name = (string?)ext.Attribute("Name");
            if (name is null || !name.StartsWith(EpExtensionPrefix, StringComparison.OrdinalIgnoreCase))
                continue;

            var publicFolder = (string?)ext.Attribute("PublicFolder");
            if (string.IsNullOrEmpty(publicFolder)) continue;

            var props = ext.Descendants().FirstOrDefault(e => e.Name.LocalName == "Properties");
            var providerName = props?.Descendants().FirstOrDefault(e => e.Name.LocalName == "ProviderName")?.Value;
            var providerPath = props?.Descendants().FirstOrDefault(e => e.Name.LocalName == "ProviderPath")?.Value;

            if (!string.IsNullOrEmpty(providerName) && !string.IsNullOrEmpty(providerPath))
                yield return (providerName!, publicFolder!, providerPath!);
        }
    }
}
