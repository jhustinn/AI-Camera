# AI-Camera: Employee Presence & Desk Occupancy Monitor

Sistem cerdas pemantauan kehadiran dan okupansi meja kerja karyawan secara *real-time* berbasis Computer Vision. Sistem ini menggabungkan pelacakan objek (YOLOv8 + ByteTrack), estimasi postur tubuh (YOLOv8-Pose), pengenalan wajah (YuNet + SFace ArcFace), logika *state machine* dengan *hysteresis*, serta dashboard analitik web berbasis FastAPI dan PostgreSQL.

```
┌─────────────────┐     ┌─────────────────────────────────────────────────────────┐
│  Kamera / RTSP  │ ──> │ YOLOv8n (Person Detection) + ByteTrack (Pelacakan ID)   │
└─────────────────┘     └───────────────────────────┬─────────────────────────────┘
                                                    │
                 ┌──────────────────────────────────┼─────────────────────────────────┐
                 ▼                                  ▼                                 ▼
      ┌──────────────────────┐          ┌──────────────────────┐          ┌──────────────────────┐
      │     ROI Kursi/Meja   │          │   Pose Estimation    │          │   Face Recognition   │
      │  Hitung Overlap Box  │          │   YOLOv8-Pose        │          │   YuNet + SFace      │
      │  Assign Meja Aktif   │          │   (DUDUK / BERDIRI)  │          │   (Cosine Sim Voting)│
      └──────────┬───────────┘          └──────────┬───────────┘          └──────────┬───────────┘
                 │                                 │                                 │
                 └─────────────────────────────────┼─────────────────────────────────┘
                                                   ▼
                                    ┌─────────────────────────────┐
                                    │    Presence State Machine   │
                                    │    (PRESENT / AWAY)         │
                                    │    Hysteresis & Grace Period│
                                    └──────────────┬──────────────┘
                                                   ▼
                                    ┌─────────────────────────────┐
                                    │    PostgreSQL Database      │
                                    │    Audit Log & Rekap Durasi │
                                    └──────────────┬──────────────┘
                                                   ▼
                                    ┌─────────────────────────────┐
                                    │    FastAPI Web Dashboard    │
                                    │    Live Stream & Laporan CSV│
                                    └─────────────────────────────┘
```

---

## Daftar Isi
- [1. Base Knowledge & Konsep Sistem](#1-base-knowledge--konsep-sistem)
  - [1.1. Deteksi & Pelacakan Orang](#11-deteksi--pelacakan-orang-yolov8--bytetrack)
  - [1.2. Pemetaan Meja Kerja (Desk Assignment & ROI)](#12-pemetaan-meja-kerja-desk-assignment--roi)
  - [1.3. Estimasi Postur Tubuh (Pose Estimation)](#13-estimasi-postur-tubuh-pose-estimation)
  - [1.4. Pengenalan Wajah (Face Recognition & Voting)](#14-pengenalan-wajah-face-recognition--voting)
  - [1.5. State Machine Presensi (Anti-Flicker & Hysteresis)](#15-state-machine-presensi-anti-flicker--hysteresis)
- [2. Kebutuhan Sistem & Arsitektur Dependensi](#2-kebutuhan-sistem--arsitektur-dependensi)
- [3. Instalasi Lingkungan Kerja](#3-instalasi-lingkungan-kerja)
- [4. Skema & Migrasi Database](#4-skema--migrasi-database)
- [5. Panduan Menjalankan Sistem](#5-panduan-menjalankan-sistem)
  - [5.1. Kalibrasi Area Kursi (ROI)](#51-kalibrasi-area-kursi-roi)
  - [5.2. Registrasi Wajah Karyawan (Enrollment)](#52-registrasi-wajah-karyawan-enrollment)
  - [5.3. Menjalankan Perekam (Recorder)](#53-menjalankan-perekam-recorder)
  - [5.4. Menjalankan Dashboard Web](#54-menjalankan-dashboard-web)
  - [5.5. Menjalankan Keduanya Sekaligus](#55-menjalankan-keduanya-sekaligus)
- [6. Konfigurasi (`config.yaml` & `.env`)](#6-konfigurasi-configyaml--env)
- [7. API & Streaming Reference](#7-api--streaming-reference)
- [8. Pengujian & Validasi](#8-pengujian--validasi)
- [9. Privasi & Kepatuhan Data](#9-privasi--kepatuhan-data)

---

## 1. Base Knowledge & Konsep Sistem

### 1.1. Deteksi & Pelacakan Orang (YOLOv8 + ByteTrack)
- Menggunakan model **YOLOv8 Nano** (`yolov8n.pt`) untuk mendeteksi objek kelas `person` (ID 0).
- Algoritma **ByteTrack** melacak pergerakan bounding box antar-frame dan memberikan `track_id` konsisten. Ini mencegah karyawan dianggap sebagai orang baru ketika berpindah posisi kecil atau terhalang sesaat.

### 1.2. Pemetaan Meja Kerja (Desk Assignment & ROI)
- Area kursi/meja didefinisikan sebagai *Region of Interest* (ROI) ternormalisasi `[x1, y1, x2, y2]` bernilai $0.0 - 1.0$.
- Hubungan antara orang dengan meja dihitung menggunakan rasio *overlap* area bawah tubuh (atau bounding box) terhadap ROI meja.
- Dilengkapi mekanisme **grace period perpindahan meja** (`desk_switch_grace_sec`) agar tidak terjadi pergantian meja secara agresif jika batas ROI saling berdekatan.

### 1.3. Estimasi Postur Tubuh (Pose Estimation)
- Menggunakan model **YOLOv8 Pose** (`yolov8n-pose.pt`) untuk mengekstrak 17 titik kerangka (*keypoints* COCO): bahu, siku, pergelangan tangan, pinggul, lutut, dan pergelangan kaki.
- Memfilter postur:
  - **DUDUK**: Rasio jarak vertikal pinggul-ke-lutut terhadap total tinggi torso berada di bawah batas tertentu (`sitting_knee_ratio`).
  - **BERDIRI**: Rasio jarak kaki ekstensi penuh (`standing_knee_ratio`).
- Dilengkapi **Skeleton Smoothing** berbasis Exponential Moving Average (EMA) agar visualisasi *stickman* tidak bergetar (*jitter*).

### 1.4. Pengenalan Wajah (Face Recognition & Voting)
- **Deteksi Wajah**: Menggunakan **YuNet** (DNN OpenCV) yang ringan dan akurat untuk wajah tampak depan maupun miring.
- **Ekstraksi Fitur**: Menggunakan **SFace** (ArcFace backbone) yang menghasilkan vektor embedding 128 dimensi bertipe `float32`.
- **Konsensus Voting (`TrackIdentityRegistry`)**: Wajah dicocokkan menggunakan *Cosine Similarity*. Untuk menghindari *false recognition*, identitas tidak langsung dikunci pada satu frame, melainkan membutuhkan akumulasi voting kecocokan (`votes_required: 3`) pada `track_id` yang sama.

### 1.5. State Machine Presensi (Anti-Flicker & Hysteresis)
Menghitung jam kerja secara naif dari deteksi frame-by-frame akan menghasilkan ribuan sesi palsu akibat *occlusion* (tertutup sesaat) atau orang menoleh. Sistem ini menggunakan *finite-state machine* dengan aturan:
- **`enter_confirm_sec` (misal 5-10 detik)**: Seseorang harus terdeteksi di kursi secara terus-menerus sebelum sesi `PRESENT` dibuka (`SESSION_OPENED`).
- **`leave_confirm_sec` (misal 4-10 detik)**: Jika orang menghilang dari deteksi sesaat, status tetap `PRESENT` untuk menoleransi *detection gap*.
- **`away_grace_sec` (misal 60 detik)**: Jika orang meninggalkan kursi lebih lama dari `leave_confirm_sec`, status berubah menjadi `AWAY` (`AWAY_START`). Jika kembali sebelum `away_grace_sec` habis, sesi dilanjutkan (`AWAY_END`).
- **`min_session_sec` (misal 300 detik)**: Sesi kerja yang total durasinya lebih singkat dari ambang batas ini akan ditandai sebagai `SESSION_DISCARDED` sehingga tidak mengotori rekap kerja.

---

## 2. Kebutuhan Sistem & Arsitektur Dependensi

- **Sistem Operasi**: Windows 10/11 atau Linux (Ubuntu 20.04+)
- **Python**: Versi 3.10 - 3.12 (Rekomendasi 3.12)
- **Database**: PostgreSQL 13 atau lebih baru
- **Akselerasi Perangkat Keras**:
  - **CPU**: Default menggunakan PyTorch CPU + OpenCV DNN (ringan di laptop kantoran).
  - **GPU CUDA (Opsional)**: NVIDIA GPU dengan CUDA 12.x untuk inferensi berkecepatan tinggi.

---

## 3. Instalasi Lingkungan Kerja

### Langkah 1: Kloning Repositori
```bash
git clone https://github.com/jhustinn/AI-Camera.git
cd AI-Camera
```

### Langkah 2: Buat Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(Di Linux / macOS: `source .venv/bin/activate`)*

### Langkah 3: Pasang Dependensi
```powershell
# Versi CPU standar (ringan)
pip install -r requirements.txt

# ATAU Versi GPU CUDA (jika memiliki NVIDIA GPU)
pip install -r requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

> **Catatan**: Bobot model (`yolov8n.pt`, `yolov8n-pose.pt`, `face_detection_yunet_2023mar.onnx`, `face_recognition_sface_2021dec.onnx`) akan otomatis diunduh saat pertama kali dijalankan dan disimpan di folder `models/`.

---

## 4. Skema & Migrasi Database

Sistem menggunakan database PostgreSQL untuk menyimpan riwayat sesi kerja, log audit kejadian, dan data biometrik wajah.

### 4.1. Konfigurasi Lingkungan (`.env`)
Salin berkas contoh `.env.example` ke `.env`:
```powershell
copy .env.example .env
```
Sesuaikan parameter koneksi database:
```ini
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/employee_presence
PRESENCE_CONFIG=config.yaml
TZ=Asia/Jakarta
```

### 4.2. Struktur Tabel Relasional (`db/schema.sql`)

| Tabel | Fungsi |
|---|---|
| `cameras` | Menyimpan daftar kamera pemantau (ID string unik, nama, dan index/URL sumber). |
| `employees` | Master data karyawan, foto, dan rata-rata vektor embedding wajah (128-dimensi bertipe `BYTEA`). |
| `desks` | Area meja kerja pada masing-masing kamera beserta koordinat ROI (`JSONB`). |
| `presence_sessions` | Sesi kehadiran kerja karyawan per meja, durasi kerja bersih (`duration_sec`), dan waktu keluar (`away_sec`). |
| `session_events` | Jejak audit peristiwa (`SESSION_OPENED`, `AWAY_START`, `AWAY_END`, `IDENTIFIED`, `SESSION_CLOSED`). |
| `live_status` | Status terkini meja kerja (real-time snapshot) yang diperbarui berkala oleh recorder. |
| `daily_summary` | SQL View untuk agregasi rekap jam kerja harian per karyawan. |

### 4.3. Menjalankan Migrasi Database

Cukup jalankan perintah CLI berikut:
```powershell
.\.venv\Scripts\python.exe run.py setup-db
```
Perintah ini akan secara otomatis:
1. Menjalankan berkas DDL [`db/schema.sql`](file:///e:/App/employee-presence/db/schema.sql) untuk membentuk seluruh tabel, indeks unik, foreign keys, dan view.
2. Melakukan sinkronisasi data kamera dari konfigurasi `config.yaml`.
3. Memperbarui daftar meja (`desks`) dan koordinat ROI ke dalam database.

*(Alternatif manual melalui psql)*:
```bash
psql -U postgres -d employee_presence -f db/schema.sql
```

---

## 5. Panduan Menjalankan Sistem

### 5.1. Kalibrasi Area Kursi (ROI)
Tentukan kotak meja/kursi pada sudut pandang kamera dengan alat interaktif:
```powershell
.\.venv\Scripts\python.exe run.py calibrate
```
- Klik dan seret mouse (*click-drag*) pada jendela preview untuk menentukan area kursi.
- Simpan konfigurasi ke `config.yaml`.

---

### 5.2. Registrasi Wajah Karyawan (Enrollment)

#### Cara A: Melalui Web Dashboard (Mudah)
1. Buka dashboard web di browser: `http://127.0.0.1:8000`.
2. Masuk ke tab **Karyawan** -> **Daftar Wajah Baru**.
3. Masukkan nama dan departemen, posisikan karyawan di depan kamera hingga sampel selesai terkumpul.

#### Cara B: Melalui CLI (Interaktif Webcam)
```powershell
.\.venv\Scripts\python.exe run.py enroll --name "Budi Santoso" --dept "Engineering"
```
Tekan `q` setelah sampel wajah mencukupi.

#### Cara C: Melalui Folder Foto
Jika sudah memiliki koleksi foto karyawan:
```powershell
.\.venv\Scripts\python.exe run.py enroll --name "Sari Wijaya" --dept "Finance" --folder .\foto\sari
```

---

### 5.3. Menjalankan Perekam (Recorder)
Recorder memproses video kamera, inferensi deteksi pose/wajah, dan mencatat kehadiran ke database:
```powershell
# Menggunakan webcam default (dengan jendela GUI OpenCV)
.\.venv\Scripts\python.exe run.py record

# Tanpa jendela GUI (cocok untuk server / latar belakang)
.\.venv\Scripts\python.exe run.py record --no-preview

# Menggunakan webcam eksternal (indeks 1) atau file video
.\.venv\Scripts\python.exe run.py record --source 1
.\.venv\Scripts\python.exe run.py record --source "D:\video\monitoring.mp4"
```

---

### 5.4. Menjalankan Dashboard Web
Jalankan server dashboard web FastAPI:
```powershell
.\.venv\Scripts\python.exe run.py serve
```
Buka browser di: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

Fitur dashboard:
- **Live Monitoring**: Tampilan status langsung tiap meja (PRESENT / AWAY / DUDUK / BERDIRI).
- **MJPEG Video Streaming**: Live feed kamera langsung di browser.
- **Rekap Harian & Per Karyawan**: Total jam kerja, waktu istirahat, dan jumlah sesi.
- **Ekspor CSV**: Unduh laporan kehadiran dengan filter tanggal.
- **Log Kejadian Real-Time**: Audit jejak masuk dan keluar karyawan.

---

### 5.5. Menjalankan Keduanya Sekaligus (Background Task di Windows PowerShell)

```powershell
$root = $PWD
# 1. Jalankan Recorder di background
Start-Process -FilePath "$root\.venv\Scripts\python.exe" `
    -ArgumentList @("run.py", "record", "--no-preview") `
    -WorkingDirectory $root -WindowStyle Hidden

# 2. Jalankan Dashboard di background
Start-Process -FilePath "$root\.venv\Scripts\python.exe" `
    -ArgumentList @("run.py", "serve") `
    -WorkingDirectory $root -WindowStyle Hidden
```

---

## 6. Konfigurasi (`config.yaml` & `.env`)

Konfigurasi utama berada di [`config.yaml`](file:///e:/App/employee-presence/config.yaml):

```yaml
timezone: Asia/Jakarta

camera:
  id: usb-cam
  name: Streaming Webcams (USB)
  source: 1                  # 0 = webcam utama, 1 = webcam kedua, atau URL RTSP
  width: 1280
  height: 720
  fps: 30
  fourcc: MJPG
  backend: dshow             # dshow (Windows DirectShow) atau any

detection:
  model: yolov8n.pt          # Model YOLOv8
  confidence: 0.35           # Ambang batas kepercayaan deteksi orang
  device: "0"                # "0" untuk GPU CUDA, "cpu" untuk CPU

face:
  enabled: true
  similarity_threshold: 0.45 # Ambang batas SFace (0.36 - 0.45)
  votes_required: 3          # Jumlah kecocokan berturut-turut sebelum identitas dikunci
  run_every_n_frames: 5      # Frekuensi pengenalan wajah (hemat beban CPU/GPU)

presence:
  enter_confirm_sec: 5       # Durasi orang harus berada di kursi sebelum status PRESENT
  leave_confirm_sec: 4       # Toleransi flicker kamera sebelum status AWAY
  away_grace_sec: 60         # Toleransi waktu meninggalkan kursi sebelum sesi ditutup
  min_session_sec: 300       # Sesi < 5 menit dianggap discarded (tidak dihitung)
  desk_overlap_ratio: 0.4    # Minimal overlap tubuh terhadap area meja

pose:
  enabled: true
  model: yolov8n-pose.pt     # Model pose stickman
  posture_labels: true       # Deteksi postur DUDUK vs BERDIRI

desks:
  - id: 1
    label: Kursi 1
    roi: [0.03, 0.10, 0.42, 1.0] # [x1, y1, x2, y2] proporsional
  - id: 2
    label: Kursi 2
    roi: [0.42, 0.10, 0.85, 1.0]

server:
  host: 127.0.0.1
  port: 8000
  stream_port: 8001          # Port MJPEG server internal
```

---

## 7. API & Streaming Reference

FastAPI menyediakan endpoint RESTful yang siap diintegrasikan dengan sistem HRIS:

| Method | Endpoint | Keterangan |
|---|---|---|
| `GET` | `/health` | Healthcheck server (`{"status": "ok"}`). |
| `GET` | `/api/status` | Snapshot status terkini seluruh meja dan FPS kamera. |
| `GET` | `/api/desks` | Daftar meja dan koordinat ROI aktif. |
| `GET` | `/api/employees` | Daftar karyawan terdaftar. |
| `GET` | `/api/sessions?day=YYYY-MM-DD` | Daftar riwayat sesi presensi. |
| `GET` | `/api/summary/daily?days=7` | Ringkasan akumulasi jam kerja harian. |
| `GET` | `/api/summary/employees` | Akumulasi durasi kerja per karyawan hari ini. |
| `GET` | `/api/events?limit=40` | Log audit aktivitas presensi terbaru. |
| `GET` | `/api/export.csv` | Unduh rekapitulasi data presensi dalam format CSV. |
| `GET` | `/stream` | Proxy MJPEG live video stream untuk web browser. |

---

## 8. Pengujian & Validasi

Proyek ini dilengkapi dengan rangkaian pengujian komprehensif:

```powershell
# Jalankan seluruh unit test (124 test)
.\.venv\Scripts\python.exe -m pytest -q

# Uji coba fungsionalitas inferensi model tanpa kamera/DB fisik
.\.venv\Scripts\python.exe run.py selftest
```

---

## 9. CCTV Multi-Kamera

Satu proses rekam per kamera + satu dashboard gabungan.
**Dokumentasi lengkap: [`docs/CCTV.md`](docs/CCTV.md).**

Ringkasan mekanisme:

- Stream diambil **langsung dari tiap kamera IP** (RTSP 554), bukan lewat NVR.
  NVR tetap merekam sendiri; sistem ini hanya "menonton".
- Format URL Dahua:
  `rtsp://USER:PASS@IP:554/cam/realmonitor?channel=1&subtype=1`
  - `channel=1` untuk kamera tunggal, `subtype=1` = sub stream 704x576 (ringan).
- Kredensial dibaca dari `.env` (`CCTV_USER` / `CCTV_PASSWORD`) lewat placeholder
  `${VAR}`, jadi tidak pernah masuk ke file config yang ter-commit.
- Template config: `config.cctv.example.yaml` -> salin jadi `config.cctv.yaml`
  (file ini di-gitignore karena berisi alamat & kredensial).

```yaml
cameras:
  - id: cctv-1
    name: D1 - meja depan
    source: rtsp://${CCTV_USER:-admin}:${CCTV_PASSWORD}@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1
    stream_port: 8001
    desks:
      - {id: 1, label: Meja 1, roi: [0.05, 0.35, 0.30, 0.95]}
  - id: cctv-2
    source: rtsp://${CCTV_USER:-admin}:${CCTV_PASSWORD}@192.168.1.109:554/cam/realmonitor?channel=1&subtype=1
    stream_port: 8002
    desks: []
```

Menjalankan:

```powershell
Copy-Item config.cctv.example.yaml config.cctv.yaml
.\.venv\Scripts\python.exe run.py --config config.cctv.yaml all
```

Dashboard: `http://127.0.0.1:8000` (grid 4 kamera + status kursi).

> `--config` ditulis **sebelum** sub-perintah.
> **Jangan menjalankan dua instance bersamaan** - batas klien RTSP kamera akan
> habis. Recorder menolak start bila port MJPEG sudah dipakai proses lain.

### Prasyarat jaringan CCTV

Jaringan CCTV terpisah dari Wi-Fi kantor, jadi PC harus tersambung **kabel LAN**
ke switch PoE dengan IP statis, dan **tidak boleh** ada konflik IP antara NVR
dan kamera. Detail lengkap + troubleshooting: [`docs/CCTV.md`](docs/CCTV.md).

### Alat bantu koneksi CCTV

```powershell
run.py rtsp-test --host 192.168.1.109 --user admin --channels 1 --skip-port-check
run.py net-scan --subnet 192.168.1
run.py watch --host 192.168.1.107 --user rtspuser --prompt --start
```

- `rtsp-test` mencoba 8 pola URL (Hikvision, Dahua, XVR generik) x semua kredensial,
  lalu melaporkan yang benar-benar menghasilkan frame.
- `net-scan` memetakan perangkat & port terbuka untuk memeriksa layer 2 / konflik IP.
- `watch` menunggu port 554 hidup, menemukan URL tiap kanal, menulis ke
  `config.cctv.yaml` (backup `.bak`), lalu menjalankan `run.py all`.

### Deployment di PC CCTV

1. Install Python di PC admin / mini PC, **bukan** di perangkat NVR/DVR.
2. `git clone`, `python -m venv .venv`, `pip install -r requirements.txt`,
   salin `.env.example` jadi `.env`.
3. Isi `DATABASE_URL`, lalu `run.py setup-db`.
4. Salin `config.cctv.example.yaml` ke `config.cctv.yaml`, isi `CCTV_USER` /
   `CCTV_PASSWORD` di `.env`, sesuaikan URL RTSP + ROI tiap meja.
5. Jalankan `run.py --config config.cctv.yaml all`, lalu aktifkan autostart
   (Task Scheduler / NSSM / systemd).

## 10. Performa

Diukur di GTX 1650, input 1280x720, kamera USB:

| Tahap | Sebelum | Sesudah |
|---|---|---|
| Deteksi + stickman | 2 model: 63 ms + 39 ms = 102 ms/frame | 1 model terpadu: **31 ms/frame** |
| Loop end-to-end | ~10 fps | **24-29 fps** |
| Tulis status DB | 1 koneksi per kursi per siklus | 1 koneksi dipakai ulang + batch |
| Angka FPS di overlay | inferensi model saja | FPS loop nyata + ms/frame |

Uji ulang: python scripts/profile_pipeline.py.

## 11. Privasi & Kepatuhan Data

Sistem ini memproses data biometrik wajah dan mencatat aktivitas kehadiran individu. Disarankan untuk:
1. Memberikan transparansi dan persetujuan tertulis (*informed consent*) kepada karyawan sesuai **UU No. 27/2022 tentang Pelindungan Data Pribadi (UU PDP)**.
2. Membatasi hak akses dashboard dan database presensi hanya untuk pihak berwenang (HRD/Admin).
3. Menerapkan kebijakan retensi data (menghapus log detail secara periodik dan menyimpan ringkasan waktu saja).
