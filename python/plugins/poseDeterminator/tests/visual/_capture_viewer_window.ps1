param(
    [string]$WindowTitlePart = "PDM Viewer",
    # ASCII-only 경로 — child PowerShell process 의 ANSI 코드페이지가 한글 경로를
    # 손상시키는 문제 회피. 호출자가 한글 경로로 복사한다.
    [string]$OutputPath = "$env:TEMP\pdm_viewer_capture.png"
)

Add-Type -AssemblyName System.Drawing

$signature = @'
using System;
using System.Runtime.InteropServices;
public class WC {
    [DllImport("user32.dll")] public static extern IntPtr FindWindow(string c, string n);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr d, uint f);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int n);
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
}
'@
Add-Type -TypeDefinition $signature

# Find viewer window by enumeration (exact title)
$proc = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like "*$WindowTitlePart*" } | Select-Object -First 1
if ($null -eq $proc) {
    Write-Error "No viewer window found containing '$WindowTitlePart'"
    exit 1
}

$hwnd = $proc.MainWindowHandle
[WC]::SetForegroundWindow($hwnd) | Out-Null
[WC]::ShowWindow($hwnd, 9) | Out-Null  # SW_RESTORE
Start-Sleep -Milliseconds 500

$rect = New-Object WC+RECT
[WC]::GetWindowRect($hwnd, [ref]$rect) | Out-Null
$width = $rect.Right - $rect.Left
$height = $rect.Bottom - $rect.Top

if ($width -le 0 -or $height -le 0) {
    Write-Error "Bad rect: w=$width h=$height"
    exit 2
}

$bmp = New-Object System.Drawing.Bitmap $width, $height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$hdc = $g.GetHdc()
# PW_RENDERFULLCONTENT = 0x2 (capture OpenGL/DirectX)
$ok = [WC]::PrintWindow($hwnd, $hdc, 0x2)
$g.ReleaseHdc($hdc)
$g.Dispose()

if (-not $ok) {
    # Fallback: copy from screen
    $bmp2 = New-Object System.Drawing.Bitmap $width, $height
    $g2 = [System.Drawing.Graphics]::FromImage($bmp2)
    $g2.CopyFromScreen($rect.Left, $rect.Top, 0, 0, [System.Drawing.Size]::new($width, $height))
    $g2.Dispose()
    $bmp2.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp2.Dispose()
    Write-Output "Saved (CopyFromScreen fallback) to $OutputPath ($width x $height)"
} else {
    $bmp.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
    Write-Output "Saved (PrintWindow) to $OutputPath ($width x $height)"
}
$bmp.Dispose()
