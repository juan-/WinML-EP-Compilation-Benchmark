using Microsoft.UI.Xaml;

namespace WinMLResNet;

public partial class App : Application
{
    private Window? _window;

    public App()
    {
        this.InitializeComponent();
        this.UnhandledException += (s, e) =>
        {
            var msg = $"[{DateTime.Now}] UNHANDLED: {e.Exception}\n";
            File.AppendAllText(Path.Combine(AppContext.BaseDirectory, "crash.log"), msg);
            e.Handled = true;
        };
    }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        try
        {
            _window = new MainWindow();
            _window.Activate();
        }
        catch (Exception ex)
        {
            File.WriteAllText(Path.Combine(AppContext.BaseDirectory, "crash.log"),
                $"[{DateTime.Now}] LAUNCH CRASH: {ex}");
        }
    }
}
