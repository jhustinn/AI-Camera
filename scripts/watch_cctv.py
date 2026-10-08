from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse  # noqa: E402

from rtsp_test import candidates, probe, tcp_check  # noqa: E402


def wait_for_port(host: str, port: int, interval: float = 10.0, timeout: float = 3600.0) -> bool:
    deadline = time.time() + timeout
    state: bool | None = None
    while time.time() < deadline:
        now = tcp_check(host, port, timeout=1.5)
        if now != state:
            stamp = time.strftime("%H:%M:%S")
            print(f"[{stamp}] port {host}:{port} {'TERBUKA' if now else 'tertutup'}", flush=True)
            state = now
        if now:
            return True
        time.sleep(interval)
    return False


def discover(host: str, user: str, password: str, channels: list[int], port: int, timeout: float) -> dict[int, str]:
    found: dict[int, str] = {}
    for channel in channels:
        print(f"\nchannel {channel}:")
        for label, url in candidates(host, user, password, channel, port):
            result = probe(url, timeout)
            if result["ok"]:
                print(f"  [OK   ] {label:28s} {result['size'][0]}x{result['size'][1]} ~{result['fps']:.0f} fps")
                found[channel] = url
                break
            print(f"  [gagal] {label:28s} {result.get('reason', '')}")
    return found


def apply_to_config(config_path: Path, found: dict[int, str]) -> int:
    """Tulis `source` tiap kamera di config berdasarkan channel RTSP yang berhasil."""
    import yaml

    if not found:
        return 0
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cameras = raw.get("cameras") or []
    if not cameras:
        return 0
    backup = config_path.with_suffix(config_path.suffix + ".bak")
    shutil.copy2(config_path, backup)

    changed = 0
    for camera in cameras:
        for index in range(1, len(cameras) + 1):
            url = found.get(index)
            if not url:
                continue
            current = str(camera.get("source", ""))
            if f"/{100 + index}" in current or f"channel={index}" in current:
                camera["source"] = url
                changed += 1
                print(f"  {camera.get('id', '?')} -> channel {index} diperbarui")
    if changed:
        config_path.write_text(
            yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        print(f"config ditulis: {config_path.name} (backup: {backup.name})")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tunggu RTSP NVR hidup, cari URL kanal, lalu tulis ke config"
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", default="")
    parser.add_argument("--prompt", action="store_true")
    parser.add_argument("--port", type=int, default=554)
    parser.add_argument("--channels", default="1,2,3,4")
    parser.add_argument("--config", default="config.cctv.yaml")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--wait", type=float, default=1800.0, help="0 = tidak menunggu")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--start", action="store_true", help="jalankan 'run.py all' setelah config diperbarui")
    args = parser.parse_args(argv)

    password = args.password
    if args.prompt:
        import getpass

        password = getpass.getpass(f"password untuk {args.user}: ")

    if args.wait > 0:
        print(f"menunggu port {args.host}:{args.port} sampai {args.wait / 60:.0f} menit...")
        if not wait_for_port(args.host, args.port, args.interval, args.wait):
            print("masih tertutup setelah batas waktu")
            return 1
    elif not tcp_check(args.host, args.port, 2.0):
        print(f"port {args.host}:{args.port} masih tertutup")
        return 1

    channels = [int(c) for c in args.channels.split(",") if c.strip().isdigit()]
    found = discover(args.host, args.user, password, channels, args.port, args.timeout)
    if not found:
        print("\ntidak ada kanal yang bisa dibuka - cek izin user & format URL NVR")
        return 1

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if args.no_write:
        print("\nURL yang berhasil:")
        for channel, url in found.items():
            print(f"  channel {channel}: {url}")
        return 0

    changed = apply_to_config(config_path, found)
    if changed == 0:
        print("\nURL berhasil ditemukan tetapi tidak ada kamera yang cocok di config.")
        print("Tambahkan 'cameras:' dengan 4 kanal (lihat config.cctv.yaml) lalu ulangi.")
        return 0

    if args.start:
        print("\nmenjalankan run.py all ...")
        import subprocess

        return subprocess.call(
            [sys.executable, str(PROJECT_ROOT / "run.py"), "--config", str(config_path), "all"],
            cwd=str(PROJECT_ROOT),
        )
    print(f"\nsiap dijalankan: run.py --config {config_path.name} all")
    return 0


if __name__ == "__main__":
    sys.exit(main())