$ErrorActionPreference = "Stop"
$log = "E:\App\employee-presence\data\lan_setup.log"
"[$(Get-Date -Format 'HH:mm:ss')] mulai" | Out-File $log -Encoding utf8

try {
    Set-NetIPInterface -InterfaceAlias "Ethernet" -Dhcp Disabled -ErrorAction Stop
    "DHCP dimatikan" | Out-File $log -Append -Encoding utf8

    Get-NetIPAddress -InterfaceAlias "Ethernet" -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Remove-NetIPAddress -Confirm:$false -ErrorAction SilentlyContinue
    "IP lama dibersihkan" | Out-File $log -Append -Encoding utf8

    New-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress "192.168.1.250" -PrefixLength 24 -ErrorAction Stop | Out-Null
    "IP statis 192.168.1.250/24 terpasang" | Out-File $log -Append -Encoding utf8

    Set-NetIPInterface -InterfaceAlias "Ethernet" -InterfaceMetric 5 -ErrorAction Stop
    "metric Ethernet = 5 (lebih prioritas dari Wi-Fi)" | Out-File $log -Append -Encoding utf8
} catch {
    "GAGAL: $($_.Exception.Message)" | Out-File $log -Append -Encoding utf8
}

Start-Sleep -Seconds 3

Get-NetIPAddress -InterfaceAlias "Ethernet" -AddressFamily IPv4 |
    Select-Object IPAddress, PrefixLength | Format-Table -AutoSize | Out-File $log -Append -Encoding utf8

Get-NetRoute -AddressFamily IPv4 -DestinationPrefix "192.168.1.0/24" |
    Select-Object InterfaceAlias, NextHop, RouteMetric | Format-Table -AutoSize | Out-File $log -Append -Encoding utf8

foreach ($ip in @("192.168.1.200", "192.168.1.14", "192.168.1.109")) {
    $ok = Test-Connection $ip -Count 2 -Quiet -ErrorAction SilentlyContinue
    "ping $ip = $ok" | Out-File $log -Append -Encoding utf8
}

foreach ($p in @(80, 554, 37777)) {
    $c = New-Object System.Net.Sockets.TcpClient
    $state = "tertutup"
    try { if ($c.ConnectAsync("192.168.1.200", $p).Wait(2000) -and $c.Connected) { $state = "TERBUKA" } } catch { }
    $c.Close()
    "192.168.1.200 port $p = $state" | Out-File $log -Append -Encoding utf8
}

"[$(Get-Date -Format 'HH:mm:ss')] selesai" | Out-File $log -Append -Encoding utf8
