# Dokumentasi CCTV — Mekanisme Kamera IP (Dahua)

Dokumen ini menjelaskan **bagaimana sistem mengambil video dari kamera CCTV IP**,
apa yang sudah terverifikasi di lapangan, dan cara troubleshoot.

---

## 1. Ringkasan Arsitektur

```
  Switch PoE CCTV
  ┌──────────────┐
  │  192.168.1.x │
  └──────┬───────┘
         │
    ┌────┴────┬─────────┬─────────┬─────────┐
    │         │         │         │         │
  Kamer 1   Kamer 2   Kamer 3   Kamer 4    NVR
  .108      .109      .110      .111      .107
  (D1)      (D2)      (D3)      (D4)
    │         │         │         │         │
    └─────────┴────┬────┴─────────┘         │
                   │  RTSP 554 (TCP)        │
                   ▼                       │
        ┌──────────────────────┐           │
        │  PC  192.168.1.250   │           │
        │  ─────────────────── │           │
        │  1 proses per kamera │           │
        │  YOLO + ByteTrack    │           │
        │  → state machine     │           │
        │  → PostgreSQL        │           │
        │  → dashboard :8000   │           │
        └──────────────────────┘
                   │
                   ▼  kabel LAN
             Router kantor 192.168.1.1
             (Wi-Fi PC laptop)
```

**Keputusan penting: stream diambil LANGSUNG dari tiap kamera, bukan lewat NVR.**

Alasannya:

| | Lewat NVR | Langsung ke kamera (dipakai) |
|---|---|---|
| Beban NVR | NVR harus meng-encode ulang 4 kanal | NVR hanya merekam sendiri |
| Ketahanan | NVR mati = semua kamera mati | 1 kamera/NVR mati tidak berpengaruh |
| Konkurensi | 1 sesi per kanal NVR | 1 sesi per kamera |
| Resolusi | Umumnya dibatasi NVR | Native 1920×1080 |
| Bitrate | Dua kali (kamera→NVR→PC) | Sekali (kamera→PC) |

NVR tetap berfungsi normal untuk merekam sendiri; sistem ini hanya **menonton**.

---

## 2. Topologi & Alamat Jaringan

| Perangkat | IP | Peran |
|---|---|---|
| Laptop (Wi-Fi) | `192.168.1.33` | Wi-Fi kantor, tidak bisa menjangkau CCTV |
| Laptop (Ethernet) | `192.168.1.250/24` | **Wajib** untuk akses CCTV |
| Router | `192.168.1.1` | Gateway |
| NVR Dahua | `192.168.1.107` | Perekam internal NVR |
| Kamera D1–D4 | `.108` – `.111` | Sumber video sistem ini |

### 2.1 Dua jaringan terpisah

Jaringan CCTV **tidak ter-uplink ke router kantor**. Karena itu:

- Laptop harus tersambung **kabel LAN** ke switch PoE CCTV.
- Adapter Ethernet butuh IP statis, bukan DHCP (tidak ada DHCP server di jaringan itu):

```powershell
Set-NetIPInterface -InterfaceAlias "Ethernet" -Dhcp Disabled
Get-NetIPAddress -InterfaceAlias "Ethernet" -AddressFamily IPv4 | Remove-NetIPAddress -Confirm:$false
New-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress 192.168.1.250 -PrefixLength 24
Set-NetIPInterface -InterfaceAlias "Ethernet" -InterfaceMetric 5
```

> **Penting:** Wi-Fi danEthernet sama-sama di `192.168.1.0/24`. Tanpa
> `InterfaceMetric 5`, Windows tetap mengirim paket ke `192.168.1.x` lewat Wi-Fi
> dan koneksi ke kamera akan **gagal** meski kabel tersambung.

Skrip siap pakai: `scripts/set_lan_static.ps1` (jalankan sebagai Administrator).

### 2.2 Konflik IP yang pernah terjadi

NVR awalnya memakai `192.168.1.108` — **sama dengan Kamera D1**. Akibatnya D1
tidak bisa dijangkau dan kanal 1 di NVR mati.

Perbaikan: pindahkan NVR ke `192.168.1.107` (`Network > TCP/IP > NIC > Modify >
Static`), lalu `Apply` + **restart NVR**.

Aturan: **alamat NVR tidak boleh sama dengan alamat kamera**.

---

## 3. Format URL RTSP (Dahua)

```
rtsp://USER:PASS@IP:554/cam/realmonitor?channel=1&subtype=1
```

| Parameter | Nilai | Arti |
|---|---|---|
| `channel` | `1` | Selalu **1** saat menarik dari kamera tunggal. Nomor ini adalah kanal NVR, bukan nomor kamera. |
| `subtype` | `0` | Main stream — 1920×1080 |
| `subtype` | `1` | Sub stream — 704×576 ✅ **default untuk sistem ini** |

Penomoran kanal NVR berbeda: untuk NVR Dahua, kanal 2 = `channel=2` (atau
`Streaming/Channels/201` untuk main / `202` untuk sub).

Alasan memakai `subtype=1`: empat main stream 1080p membebani GTX 1650 4GB.
Sub stream 704×576 cukup untuk deteksi okupansi dan postural, dengan hasil
terukur **22–26 fps per kamera** (4 kamera simultan).

Naikkan ke `subtype=0` hanya bila perlu kualitas tinggi (mis. enroll wajah),
dan kurangi jumlah kamera aktif bila FPS turun.

---

## 4. Autentikasi

Kamera Dahua memakai **HTTP Digest** untuk RTSP, Basic untuk web.

Terdapat **dua lapisan akun yang berbeda** dan sering tertukar:

| Arah | Akun | Keterangan |
|---|---|---|
| NVR → Kamera | akun **kamera** (`admin` + password kamera) | Diisi di `Camera List > Modify` pada NVR |
| PC/NVR Client → NVR | akun **NVR** (`rtspuser` + password NVR) | Untuk menarik stream dari NVR |
| PC → Kamera (langsung) | akun **kamera** (`admin` + password kamera) | Dipakai sistem ini |

> ⚠️ Mengisi akun NVR (`rtspuser`) di kolom kamera pada NVR **pasti ditolak**.
> Itu bukan bug — memang dua database user yang terpisah.

### 4.1 Menyimpan kredensial

Repo ini **publik**, jadi kredensial asli tidak boleh masuk file config.
`src/presence/config.py` melakukan ekspansi `${VAR}` dan `${VAR:-default}`
sebelum YAML di-parse.

Buat `.env` (sudah masuk `.gitignore`):

```
CCTV_USER=admin
CCTV_PASSWORD=password-kamera
```

Lalu di `config.cctv.yaml`:

```yaml
source: rtsp://${CCTV_USER:-admin}:${CCTV_PASSWORD}@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1
```

`config.cctv.yaml` juga di-`.gitignore`; yang di-commit hanya
`config.cctv.example.yaml` (tanpa kredensial).

### 4.2 Lockout

Kamera Dahua mengunci IP sementara setelah beberapa kali login gagal. Mengira
kamera "mati" padahal sebenarnya terkunci. Tunggu beberapa menit sebelum
mencoba password lain.

---

## 5. Alat Bantu

| Perintah | Fungsi |
|---|---|
| `run.py cameras` | Daftar kamera & uji sumber RTSP |
| `run.py rtsp-test --host <ip> --user <u> --password <p> --channels 1` | Coba 8 pola URL per kanal, laporkan mana yang menghasilkan frame |
| `run.py net-scan --subnet 192.168.1` | Peta perangkat + port terbuka di jaringan CCTV |
| `run.py watch --host <ip> ... --start` | Tunggu port RTSP hidup, cari URL, tulis config, jalankan |
| `scripts/port_scan.py --ip <ip> --banner` | Scan 1–10000 + baca banner HTTP |
| `scripts/profile_pipeline.py` | Ukur ms/frame model deteksi |
| `scripts/restart_all.ps1` | Stop semua instance, pastikan port bebas, start ulang |

Contoh:

```powershell
.venv\Scripts\python.exe run.py --config config.cctv.yaml rtsp-test `
    --host 192.168.1.109 --user admin --channels 1 --skip-port-check
```

---

## 6. Menjalankan

```powershell
Copy-Item config.cctv.example.yaml config.cctv.yaml   # sekali saja
# isi .env dengan CCTV_USER / CCTV_PASSWORD
.venv\Scripts\python.exe run.py --config config.cctv.yaml all
```

`run.py all` menyalakan **satu proses per kamera** + satu dashboard:

| ID | Port MJPEG | Dashboard |
|---|---|---|
| cctv-1 | 8001 | http://127.0.0.1:8000 |
| cctv-2 | 8002 | |
| cctv-3 | 8003 | |
| cctv-4 | 8004 | |

Semua `--config` harus ditulis **sebelum** sub-perintah
(`run.py --config X all`), bukan sesudahnya.

> **Jangan pernah menjalankan dua instance bersamaan.** Batas klien RTSP
> kamera akan habis, semua stream mati, dan tampilan dashboard tetap normal
> padahal deteksi sudah berhenti. Recorder sekarang menolak start bila port
> MJPEG sudah dipakai proses lain.

---

## 7. Pipeline Deteksi (per kamera)

```
frame RTSP 704x576
  → deteksi orang + keypoint (yolov8n-pose, imgsz 640, CUDA)
  → ByteTrack  → track_id stabil
  → smoothing keypoint + validasi-member
  → klasifikasi postur (duduk / berdiri)
  → DeskAssigner (ROI + hampiran torso, hysteresis)
  → state machine (enter/leave/away + grace)
  → PostgreSQL (presence_sessions, session_events, live_status)
```

Mode `detection.unified: true` memakai **satu** model untuk deteksi + tracking
+ keypoint, bukan dua model terpisah. Terukur 31 ms/frame vs 102 ms/frame,
.loop 24–30 fps vs ~10 fps.

### 7.1 Hysteresis state machine

Transisi status memerlukan konfirmasi waktu agar tidak berfluktuasikan:

| Config | Default | Fungsi |
|---|---|---|
| `enter_confirm_sec` | 8 | Orang harus bertahan agar PRESENT |
| `leave_confirm_sec` | 6 | Kehilangan harus bertahan agar AWAY |
| `away_grace_sec` | 90 | Istirahat singkat tidak menutup sesi |
| `min_session_sec` | 300 | Sesi lebih dari 5 menit dianggap nyata |
| `desk_switch_grace_sec` | 10 | Jeda berpindah meja |

Tanpa hysteresis ini status berkedip PRESENT/AWAY setiap kali seseorang
bergeser kursi.

---

## 8. Dashboard

`http://127.0.0.1:8000` menampilkan:

- **Pratinjau Kamera** — grid 4 kamera MJPEG, masing-masing memakai
  `stream_url` miliknya (bukan satu URL global).
- **Status Real-time per Kursi** — status, durasi, fps, track_id.
- **Sesi Kehadiran** — tabel harian + export CSV.
- **Rekap Harian / Durasi per Karyawan** — Chart.js.
- **Pendaftaran Wajah** — enroll lewat browser.
- **Pemetaan Karyawan ke Kursi**, **Aktivitas Terbaru**.

---

## 9. Troubleshooting

### Gejala: `ERR_CONNECTION_REFUSED` atau dashboard "memuat data..."

**Penyebab 1 — JS dashboard rusak.**
Gejalanya: *seluruh* halaman tidak bergerak, badge tetap
`recorder:connecting`, kartu kursi kosong stuck di "memuat data...".

Periksa sintaksnya:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_dashboard.py -v
```

Test `test_dashboard_inline_js_is_valid_syntax` mem-extract blok `<script>` dan
memvalidasi dengan `node --check`. Backtick yang hilang pada template literal
mematikan seluruh skrip secara diam-diam.

**Penyebab 2 — proses mati.**
Cek jumlah proses dan port:

```powershell
(Get-Process python).Count
8000..8004 | ForEach-Object { "$_ -> " + (Get-NetTCPConnection -LocalPort $_ -State Listen -EA SilentlyContinue).Count }
```

**Penyebab 3 — instance ganda.**
`data/serve.log` akan memuat
`[Errno 10048] ... only one usage of each socket address`. Stop semua proses
lalu start ulang **satu** instance.

> Catatan: thread MJPEG tetap melayani frame **basi** setelah loop deteksi
> berhenti, sehingga `curl /stream` masih balas HTTP 200 padahal rekaman sudah
> mati. Verifikasi lewat `data/record_<id>.log`, bukan HTTP status.

### Gejala: semua kamera mati serentak

Selalu periksa dulu:

```powershell
Get-NetAdapter -Name Ethernet | Select-Object Status, LinkSpeed
```

Kabel RJ45 ke switch PoE mudah goyah. `Status: Disconnected` berarti **link
physis hilang** — bukan masalah kamera, NVR, atau software.Gejalanya khas:
semua stream drop di detik yang sama persis.

### Gejala: kamera tidak ditemukan

```powershell
.venv\Scripts\python.exe run.py net-scan --subnet 192.168.1
```

Kalau perangkat tidak muncul sama sekali → masalah layer 2 (kabel/switch).
Kalau muncul tapi port 554 tertutup → RTSP nonaktif atau IP konflik.

### Gejala: `wrong username or password` dari NVR

Akun NVR (`rtspuser`) tidak berlaku untuk kolom kamera. Isi **akun kamera**
(`admin` + password kamera).

### Gejala: kamera membeku 1–2 menit lalu `end of file reached`

Riwayat bug: `_is_file_source()` memakai `not source.isdigit()`, sehingga
**setiap URL RTSP dianggap file video**. Begitu satu read gagal, recorder
mencetak "end of file" lalu berhenti permanen tanpa mencoba reconnect.

Perbaikan: klasifikasi sumber dipisah dengan benar
(`tests/test_recorder_source.py`), dan `_reconnect()` sekarang looping dengan
backoff sampai link kembali, bukan menyerah setelah satu percobaan.

---

## 10. Pemangkasan

Opsi bila resource terbatas:

| Opsi | Dampak |
|---|---|
| `subtype=1` (sub stream) ✅ sudah dipakai | 4 kamera @ 26 fps di GTX 1650 4GB |
| `imgsz: 512` | ~30% lebih cepat, keypoint kurang presisi |
| `detection.unified: false` | **jangan** — 3× lebih lambat |
| Kurangi `max_width` | Hemat kecil, akurasi sedikit turun |
| 2 kamera aktif | ~2× lebih banyak fps per kamera |

---

## 11. Catatan Keamanan

- Kredensial kamera **tidak boleh** masuk commit (repo publik). Pakai `.env`.
- `config.cctv.yaml` di-`.gitignore`; template yang di-commit adalah
  `config.cctv.example.yaml`.
- Dashboard tanpa autentikasi — bind ke `127.0.0.1` atau pasang reverse proxy
  bila perlu diakses dari jaringan lain.
- RekamanASNtakes presence dan wajah karyawan: pastikan ada pemberitahuan dan
  persetujuan sesuai kebijakan data pribadi perusahaan.

---

## 12. Ringkasan Troubleshooting Cepat

| Gejala | Cause | Solusi |
|---|---|---|
| Halaman beku "memuat data" | JS template rusak | `pytest tests/test_dashboard.py` |
| Semua kamera mati serentak | Kabel LAN lepas | Cek `Get-NetAdapter` |
| Kanal 1 tidak ada | NVR & kamera berebut `.108` | Pindahkan NVR ke `.107` |
| `wrong username` di NVR | Akun NVR ≠ akun kamera | Pakai akun kamera |
| Stream berhenti permanen | Reconnect gagal | Sudah diperbaiki; update kode |
| `Errno 10048` | Instance ganda | Stop semua, start 1 |
| Kamera tidak terlihat | Layer 2 / IP konflik | `run.py net-scan` |
