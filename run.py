from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse  # noqa: E402


def cmd_setup_db(args: argparse.Namespace) -> int:
    from presence import db
    from presence.config import load_config

    cfg = load_config(args.config)
    db.apply_schema()
    camera_id = db.ensure_camera(cfg.camera.id, cfg.camera.name, str(cfg.camera.source))
    db.sync_desks(
        camera_id,
        [{"label": d.label, "roi": d.roi, "employee_id": d.employee_id} for d in cfg.desks],
    )
    print(f"camera_id={camera_id} ({cfg.camera.id})")
    print(f"desks={len(cfg.desks)}")
    for row in db.list_desks():
        print(f"  #{row['id']} {row['label']} roi={row['roi']} employee={row['employee_name'] or '-'}")
    print(f"employees={len(db.list_employees())}")
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    from presence.recorder import main as recorder_main

    argv: list[str] = []
    if args.config:
        argv += ["--config", args.config]
    if args.source is not None:
        argv += ["--source", args.source]
    if args.camera_id:
        argv += ["--camera-id", args.camera_id]
    if args.backend:
        argv += ["--backend", args.backend]
    if args.no_preview:
        argv.append("--no-preview")
    if args.no_save_frames:
        argv.append("--no-save-frames")
    return recorder_main(argv)


def cmd_record_all(args: argparse.Namespace) -> int:
    """Jalankan satu proses rekam per kamera + satu dashboard."""
    import subprocess
    import sys as _sys
    import time as _time

    from presence.config import load_config

    cfg = load_config(args.config)
    root = Path(__file__).resolve().parent
    children = []
    for channel in cfg.channels:
        argv = [_sys.executable, str(root / "run.py")]
        if args.config:
            argv += ["--config", args.config]
        argv += ["record", "--camera-id", channel.id, "--no-preview"]
        log = root / "data" / f"record_{channel.id}.log"
        handle = open(log, "a", encoding="utf-8", buffering=1)
        print(f"mulai rekam: {channel.id} -> {channel.source} (log: {log.name})")
        children.append(
            (
                channel.id,
                subprocess.Popen(argv, cwd=str(root), stdout=handle, stderr=subprocess.STDOUT),
                handle,
            )
        )
        _time.sleep(1.0)

    serve_argv = [_sys.executable, str(root / "run.py")]
    if args.config:
        serve_argv += ["--config", args.config]
    serve_argv.append("serve")
    serve_log = root / "data" / "serve.log"
    serve_handle = open(serve_log, "a", encoding="utf-8", buffering=1)
    print(f"dashboard: http://{cfg.server.host}:{cfg.server.port}")
    children.append(("__dashboard__", subprocess.Popen(serve_argv, cwd=str(root), stdout=serve_handle, stderr=subprocess.STDOUT), serve_handle))

    print(f"{len(children)} proses berjalan. Ctrl+C untuk menghentikan semua.")
    try:
        while True:
            alive = [(name, proc) for name, proc, _h in children if proc.poll() is None]
            for name, proc, _h in children:
                if proc.poll() is not None and name != "__dashboard__":
                    print(f"proses '{name}' berhenti (kode {proc.returncode})")
            if not alive:
                break
            _time.sleep(2)
    except KeyboardInterrupt:
        print("menghentikan semua proses...")
    finally:
        for _name, proc, _handle in children:
            if proc.poll() is None:
                proc.terminate()
        for _name, proc, handle in children:
            try:
                proc.wait(timeout=8)
            except Exception:  # noqa: BLE001
                proc.kill()
            handle.close()
    return 0


def cmd_rtsp_test(args: argparse.Namespace) -> int:
    import rtsp_test

    extra = ["--also", args.also] if args.also else []
    if args.prompt:
        extra.append("--prompt")
    return rtsp_test.main([
        "--host", args.host,
        "--user", args.user,
        "--password", args.password,
        *extra,
        "--port", str(args.port),
        "--channels", args.channels,
        "--timeout", str(args.timeout),
    ])


def cmd_watch(args: argparse.Namespace) -> int:
    import watch_cctv

    extra = ["--prompt"] if args.prompt else []
    if args.start:
        extra.append("--start")
    if args.no_write:
        extra.append("--no-write")
    return watch_cctv.main([
        "--host", args.host,
        "--user", args.user,
        "--password", args.password,
        *extra,
        "--port", str(args.port),
        "--channels", args.channels,
        "--config", args.config or "config.cctv.yaml",
        "--interval", str(args.interval),
        "--wait", str(args.wait),
    ])


def cmd_net_scan(args: argparse.Namespace) -> int:
    import net_scan

    argv = ["--subnet", args.subnet, "--ports", args.ports]
    if args.only_known:
        argv += ["--only-known", args.only_known]
    return net_scan.main(argv)


def cmd_cameras(args: argparse.Namespace) -> int:
    from presence.config import load_config

    cfg = load_config(args.config)
    if not cfg.channels:
        print("tidak ada kamera di config")
        return 1
    print(f"{len(cfg.channels)} kamera:\n")
    for index, channel in enumerate(cfg.channels):
        source = channel.source if not isinstance(channel.source, str) else _mask_credentials(channel.source)
        print(f"  {index + 1}. id={channel.id}")
        print(f"     nama  : {channel.name}")
        print(f"     sumber: {source}")
        print(f"     video : {channel.width}x{channel.height} @ {channel.fps}fps (fourcc {channel.fourcc})")
        print(f"     stream: http://{cfg.server.stream_host}:{channel.stream_port}/stream")
        print(f"     kursi : {len(channel.desks)} -> " + ", ".join(d.label for d in channel.desks))
    print("\n Jalankan salah satu: run.py record --camera-id <id>")
    print(" Jalankan semua + dashboard : run.py all")
    return 0


def _mask_credentials(url: str) -> str:
    if "@" not in url or "//" not in url:
        return url
    scheme, rest = url.split("//", 1)
    credentials, host = rest.split("@", 1)
    user = credentials.split(":")[0]
    return f"{scheme}//{user}:***@{host}"


def cmd_serve(args: argparse.Namespace) -> int:
    import os as _os

    import uvicorn

    from presence.config import load_config

    # routes/main membaca PRESENCE_CONFIG saat diimpor -> set sebelum import.
    if args.config:
        _os.environ["PRESENCE_CONFIG"] = args.config
    from presence.web.main import create_app

    cfg = load_config(args.config)
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port
    print(f"dashboard: http://{host}:{port}  (config: {args.config or 'config.yaml'})")
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    import calibrate_roi

    argv: list[str] = []
    if args.config:
        argv += ["--config", args.config]
    if args.source is not None:
        argv += ["--source", args.source]
    return calibrate_roi.main(argv)


def cmd_enroll(args: argparse.Namespace) -> int:
    import enroll_face

    argv = ["--name", args.name, "--backend", args.backend]
    if args.employee_no:
        argv += ["--employee-no", args.employee_no]
    if args.dept:
        argv += ["--dept", args.dept]
    if args.folder:
        argv += ["--folder", args.folder]
    if args.video:
        argv += ["--video", args.video]
        if args.start_frame:
            argv += ["--start-frame", str(args.start_frame)]
    if args.count:
        argv += ["--count", str(args.count)]
    if args.config:
        argv += ["--config", args.config]
    return enroll_face.main(argv)


def cmd_status(args: argparse.Namespace) -> int:
    from presence import db
    from presence.config import load_config

    cfg = load_config(args.config)
    rows = db.fetch_live_status()
    print(f"timezone={cfg.timezone}")
    print(f"live_status rows={len(rows)}")
    for row in rows:
        print(
            f"  {row['label']}: {row['status']} employee={row['employee_name'] or '-'} "
            f"duration={row['duration_sec']}s fps={row['fps']}"
        )
    open_sessions = db.fetch_sessions(cfg.timezone, limit=20)
    print(f"sesi terbaru={len(open_sessions)}")
    for row in open_sessions[:10]:
        print(
            f"  #{row['id']} {row['desk_label']} {row['employee_name'] or '-'} "
            f"{row['status']} dur={row['duration_sec']}s away={row['away_sec']}s"
        )
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    import selftest

    return selftest.main(list(getattr(args, "rest", []) or []))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Employee Presence Monitor")
    parser.add_argument("--config", default=None, help="path ke config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup-db", help="buat tabel & seed kamera/kursi").set_defaults(func=cmd_setup_db)

    record = sub.add_parser("record", help="jalankan pencamatan real-time")
    record.add_argument("--source", default=None)
    record.add_argument("--backend", default="sface", choices=["sface", "insightface"])
    record.add_argument("--no-preview", action="store_true")
    record.add_argument("--no-save-frames", action="store_true")
    record.add_argument("--camera-id", default=None, help="pilih kamera dari daftar cameras:")
    record.set_defaults(func=cmd_record)

    record_all = sub.add_parser("all", help="rekam semua kamera + dashboard")
    record_all.set_defaults(func=cmd_record_all)

    cameras = sub.add_parser("cameras", help="lihat daftar kamera & uji sumber RTSP")
    cameras.set_defaults(func=cmd_cameras)

    rtsp = sub.add_parser("rtsp-test", help="coba pola URL RTSP NVR & tampilkan yang berhasil")
    rtsp.add_argument("--host", required=True)
    rtsp.add_argument("--user", default="admin")
    rtsp.add_argument("--password", default="")
    rtsp.add_argument("--also", default="", help="kredensial tambahan: user2:pass2")
    rtsp.add_argument("--prompt", action="store_true", help="input password interaktif")
    rtsp.add_argument("--port", type=int, default=554)
    rtsp.add_argument("--channels", default="1")
    rtsp.add_argument("--timeout", type=float, default=8.0)
    rtsp.set_defaults(func=cmd_rtsp_test)

    scan = sub.add_parser("net-scan", help="petakan perangkat & port terbuka di jaringan CCTV")
    scan.add_argument("--subnet", default="192.168.1")
    scan.add_argument("--ports", default="554,80,37777,8000,34567,8080")
    scan.add_argument("--only-known", default="")
    scan.set_defaults(func=cmd_net_scan)

    watch = sub.add_parser(
        "watch", help="tunggu RTSP NVR hidup, cari URL kanal, tulis config, lalu jalankan"
    )
    watch.add_argument("--host", required=True)
    watch.add_argument("--user", required=True)
    watch.add_argument("--password", default="")
    watch.add_argument("--prompt", action="store_true")
    watch.add_argument("--port", type=int, default=554)
    watch.add_argument("--channels", default="1,2,3,4")
    watch.add_argument("--config", default=None)
    watch.add_argument("--interval", type=float, default=10.0)
    watch.add_argument("--wait", type=float, default=1800.0)
    watch.add_argument("--no-write", action="store_true")
    watch.add_argument("--start", action="store_true")
    watch.set_defaults(func=cmd_watch)

    serve = sub.add_parser("serve", help="jalankan dashboard web")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.set_defaults(func=cmd_serve)

    calibrate = sub.add_parser("calibrate", help="kalibrasi area kursi")
    calibrate.add_argument("--source", default=None)
    calibrate.set_defaults(func=cmd_calibrate)

    enroll = sub.add_parser("enroll", help="daftarkan wajah karyawan")
    enroll.add_argument("--name", required=True)
    enroll.add_argument("--employee-no", default=None)
    enroll.add_argument("--dept", default=None)
    enroll.add_argument("--folder", default=None)
    enroll.add_argument("--video", default=None)
    enroll.add_argument("--count", type=int, default=10)
    enroll.add_argument("--start-frame", type=int, default=0)
    enroll.add_argument("--backend", default="sface", choices=["sface", "insightface"])
    enroll.set_defaults(func=cmd_enroll)

    sub.add_parser("status", help="lihat status & sesi terbaru").set_defaults(func=cmd_status)
    selftest = sub.add_parser("selftest", help="cek model tanpa kamera/DB")
    selftest.add_argument("rest", nargs=argparse.REMAINDER)
    selftest.set_defaults(func=cmd_selftest)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())