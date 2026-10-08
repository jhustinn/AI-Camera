$ErrorActionPreference = "Continue"
$log = "E:\App\employee-presence\data\cleanup.log"

Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*employee-presence*" } |
    ForEach-Object {
        "membunuh PID $($_.ProcessId): $($_.CommandLine)" | Out-File $log -Append -Encoding utf8
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

Start-Sleep -Seconds 4
"--- sisa python proyek ---" | Out-File $log -Append -Encoding utf8
$left = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*employee-presence*" }
if ($left) { $left | ForEach-Object { $_.ProcessId } | Out-File $log -Append -Encoding utf8 }
else { "bersih" | Out-File $log -Append -Encoding utf8 }

"--- port ---" | Out-File $log -Append -Encoding utf8
foreach ($p in 8000, 8001, 8002, 8003, 8004) {
    $n = (Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue | Measure-Object).Count
    "port $p listener=$n" | Out-File $log -Append -Encoding utf8
}
"selesai" | Out-File $log -Append -Encoding utf8
