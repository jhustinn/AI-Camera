from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path
from urllib.parse import quote

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|timeout;5000000|stimeout;5000000",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse  # noqa: E402

import cv2  # noqa: E402


def candidates(host: str, user: str, password: str, channel: int, port: int = 554) -> list[tuple[str, str]]:
    """Pola URL RTSP yang umum dipakai NVR (Hikvision, Dahua, XVR generik)."""
    auth = f"{quote(user, safe='')}:{quote(password, safe='')}@"
    base = f"rtsp://{auth}{host}:{port}"
    ch1 = channel + 100
    return [
        ("Hikvision / Dahua (channel)", f"{base}/Streaming/Channels/{ch1}"),
        ("Hikvision + transport tcp", f"{base}/Streaming/Channels/{ch1}?transportmode=tcp"),
        ("Dahua realmonitor main", f"{base}/cam/realmonitor?channel={channel}&subtype=0"),
        ("Dahua realmonitor sub", f"{base}/cam/realmonitor?channel={channel}&subtype=1"),
        ("XVR generik stream=0", f"{base}/user={user}_channel={channel}_stream=0.sdp?real_stream"),
        ("XVR generik stream=1", f"{base}/user={user}_channel={channel}_stream=1.sdp?real_stream"),
        ("Path umum /live", f"{base}/live/ch{channel}"),
        ("Path umum /ch", f"{base}/ch{channel}/0"),
    ]


def tcp_check(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:  # noqa: BLE001
        return False


def host_of(url: str) -> tuple[str, int] | None:
    try:
        rest = url.split("://", 1)[1]
        authority = rest.split("@", 1)[-1]
        authority = authority.split("/", 1)[0]
        authority = authority.split("?", 1)[0]
        host, _, port = authority.partition(":")
        return host, int(port or 554)
    except Exception:  # noqa: BLE001
        return None


def probe(url: str, timeout: float, want_frames: int = 5, port_timeout: float = 2.0) -> dict:
    target = host_of(url)
    if target and not tcp_check(target[0], target[1], port_timeout):
        return {"ok": False, "reason": f"port {target[1]} tertutup"}
    started = time.perf_counter()
    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout * 1000))
    capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(timeout * 1000))
    if not capture.isOpened():
        capture.release()
        return {"ok": False, "reason": "tidak terbuka (kredensial/URL salah)"}
    frames = 0
    size = None
    last = time.perf_counter()
    while frames < want_frames and time.perf_counter() - started < timeout:
        ok, frame = capture.read()
        if not ok:
            break
        size = (frame.shape[1], frame.shape[0])
        frames += 1
    fps = frames / max(1e-6, time.perf_counter() - last)
    capture.release()
    return {
        "ok": frames > 0,
        "frames": frames,
        "size": size,
        "fps": fps,
        "seconds": round(time.perf_counter() - started, 2),
        "reason": "" if frames else "terbuka tapi tidak ada frame (channel kosong/terblokir)",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cek RTSP NVR & temukan pola URL yang benar")
    parser.add_argument("--host", required=True, help="IP NVR, misal 192.168.1.14")
    parser.add_argument("--user", default="admin")
    parser.add_argument("--password", default="")
    parser.add_argument("--also", default="", help="kredensial tambahan: user2:pass2,user3:pass3")
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="minta password secara interaktif (tidak tersimpan di riwayat shell)",
    )
    parser.add_argument("--port", type=int, default=554)
    parser.add_argument("--channels", default="1", help="daftar channel, misal 1,2,3,4")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--skip-port-check", action="store_true")
    parser.add_argument("--force", action="store_true", help="tetep coba walau port tertutup")
    args = parser.parse_args(argv)

    credentials = [(args.user, args.password)]
    for item in args.also.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        name, secret = item.split(":", 1)
        if (name, secret) not in credentials:
            credentials.append((name, secret))
    if args.prompt:
        import getpass

        credentials = [(name, getpass.getpass(f"password untuk {name}: ")) for name, _p in credentials]

    print(f"target {args.host}:{args.port}  kredensial: " + ", ".join(user for user, _p in credentials))
    open_now = True
    if not args.skip_port_check:
        open_now = tcp_check(args.host, args.port)
        print(f"  port TCP terbuka : {open_now}")
        if not open_now:
            print(
                "\n  Porta tertutup -> layanan NVR belum aktif (user baru tidak akan membantu dulu).\n"
                "   1. NETWORK > Port: tekan Apply, lalu restart/power-cycle NVR\n"
                "   2. cek http://<ip-nvr> dari browser PC ini\n"
                "   3. setelah port terbuka, ulangi perintah ini\n"
                "   Catatan: halaman 'Register'/P2P tidak diperlukan untuk sistem ini"
            )
            if not args.force:
                return 1

    working: list[tuple[int, str, str]] = []
    for raw_channel in args.channels.split(","):
        raw_channel = raw_channel.strip()
        if not raw_channel.isdigit():
            continue
        channel = int(raw_channel)
        print(f"\nchannel {channel}:")
        for user, password in credentials:
            for label, url in candidates(args.host, user, password, channel, args.port):
                result = probe(url, args.timeout)
                status = "OK" if result["ok"] else "gagal"
                detail = ""
                if result["ok"]:
                    detail = f" {result['size'][0]}x{result['size'][1]} ~{result['fps']:.0f} fps"
                    working.append((channel, f"{user} / {label}", url))
                elif result.get("reason"):
                    detail = f" ({result['reason']})"
                print(f"  [{status:5s}] {user:10s} {label:26s}{detail}")
            if working:
                break

    if working:
        print("\nURL yang bisa dipakai (tempel ke config.cctv.yaml):")
        for channel, label, url in working:
            print(f"  - source: {url}   # channel {channel} ({label})")
    else:
        print("\nBelum ada URL yang berhasil. Periksa: RTSP aktif? kredensial benar?")
    return 0 if working else 1


if __name__ == "__main__":
    sys.exit(main())