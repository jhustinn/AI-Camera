$ErrorActionPreference = "Continue"
$root = "E:\App\employee-presence"
$log = "$root\data\restart.log"
$py = "$root\.venv\Scripts\python.exe"

function Say($m) {
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $m"
    $line | Out-File $log -Append -Encoding utf8
    Write-Host $line
}

"=== RESTART $(Get-Date) ===" | Out-File $log -Encoding utf8

Say "1/4  menghentikan semua python..."
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='python3.12.exe'" |
    Where-Object { $_.CommandLine -like "*employee-presence*" }
if ($procs) {
    foreach ($p in $procs) {
        Say "    stop PID $($p.ProcessId)"
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
} else {
    Say "    (tidak ada proses proyek)"
}

Start-Sleep -Seconds 5

$left = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='python3.12.exe'" |
    Where-Object { $_.CommandLine -like "*employee-presence*" }
Say "    sisa: $(if ($left) { ($left.ProcessId -join ',') } else { 'bersih' })"

Say "2/4  memastikan port 8000-8004 bebas..."
foreach ($port in 8000, 8001, 8002, 8003, 8004) {
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($conn) {
        Say "    port $port masih dipegang PID $($conn.OwningProcess) - dipaksa stop"
        Stop-Process -Id ($conn.OwningProcess | Select-Object -Unique) -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Seconds 4

Say "3/4  memeriksa link Ethernet & kamera..."
$ad = Get-NetAdapter -Name "Ethernet" -ErrorAction SilentlyContinue
Say "    Ethernet: $($ad.Status) $($ad.LinkSpeed)"
foreach ($ip in @("192.168.1.107", "192.168.1.108", "192.168.1.109", "192.168.1.110", "192.168.1.111")) {
    $c = New-Object System.Net.Sockets.TcpClient
    $s = "tutup"
    try { if ($c.ConnectAsync($ip, 554).Wait(3000) -and $c.Connected) { $s = "554 BUKA" } } catch { }
    $c.Close()
    Say "    $ip RTSP $s"
}

Say "4/4  menjalankan run.py --config config.cctv.yaml all ..."
Remove-Item "$root\data\record_cctv-*.log" -Force -ErrorAction SilentlyContinue
$p = Start-Process -FilePath $py -ArgumentList "run.py", "--config", "config.cctv.yaml", "all" `
     -WorkingDirectory $root -WindowStyle Hidden -PassThru
Say "sistem berjalan, parent PID $($p.Id)"
Say "dashboard: http://127.0.0.1:8000"
Say "=== SELESAI ==="
