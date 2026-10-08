$ErrorActionPreference = "Continue"
$log = "E:\App\employee-presence\data\lan_watch.log"
$root = "E:\App\employee-presence"
$py = "$root\.venv\Scripts\python.exe"

function Say($msg) { "[$(Get-Date -Format 'HH:mm:ss')] $msg" | Out-File $log -Append -Encoding utf8 }

Say "menunggu link Ethernet..."
$deadline = (Get-Date).AddMinutes(90)
while ((Get-Date) -lt $deadline) {
    $ad = Get-NetAdapter -Name "Ethernet" -ErrorAction SilentlyContinue
    if ($ad -and $ad.Status -eq "Up") {
        Say "LINK UP ($($ad.LinkSpeed))"
        break
    }
    Start-Sleep -Seconds 5
}

if ((Get-Date) -ge $deadline) { Say "batal: link tidak naik dalam 90 menit"; exit 1 }

Start-Sleep -Seconds 10
$ip = (Get-NetIPAddress -InterfaceAlias "Ethernet" -AddressFamily IPv4 |
       Where-Object { $_.IPAddress -notlike "169.254*" } | Select-Object -First 1).IPAddress
Say "IP laptop di LAN: $ip"

foreach ($t in @("192.168.1.107","192.168.1.108","192.168.1.109","192.168.1.110","192.168.1.111")) {
    $c = New-Object System.Net.Sockets.TcpClient
    $s = "tutup"
    try { if ($c.ConnectAsync($t, 554).Wait(3000) -and $c.Connected) { $s = "554 BUKA" } } catch { }
    $c.Close()
    $p = Test-Connection $t -Count 1 -Quiet -ErrorAction SilentlyContinue
    Say "$t : ping=$p | RTSP $s"
}

Say "menjalankan run.py --config config.cctv.yaml all ..."
$p = Start-Process -FilePath $py -ArgumentList "run.py","--config","config.cctv.yaml","all" `
     -WorkingDirectory $root -WindowStyle Hidden -PassThru
Say "sistem berjalan (PID $($p.Id))"
