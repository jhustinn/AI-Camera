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
    if args.backend:
        argv += ["--backend", args.backend]
    if args.no_preview:
        argv.append("--no-preview")
    if args.no_save_frames:
        argv.append("--no-save-frames")
    return recorder_main(argv)


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from presence.config import load_config
    from presence.web.main import create_app

    cfg = load_config(args.config)
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port
    print(f"dashboard: http://{host}:{port}")
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
    record.set_defaults(func=cmd_record)

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