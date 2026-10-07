from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse  # noqa: E402
import logging  # noqa: E402
from typing import Any  # noqa: E402

import cv2  # noqa: E402
import yaml  # noqa: E402

from presence import db  # noqa: E402
from presence.config import load_config  # noqa: E402

LOGGER = logging.getLogger("presence.calibrate")


def parse_source(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    if value.isdigit():
        return int(value)
    return value


def grab_frame(source: Any, warmup: int = 10) -> cv2.VideoCapture | None:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        LOGGER.error("tidak bisa membuka sumber %s", source)
        return None
    for _ in range(warmup):
        ok, frame = capture.read()
        if not ok:
            break
    return capture


def draw(frame, regions, highlight: int | None = None):
    height, width = frame.shape[:2]
    for index, (label, roi) in enumerate(regions):
        x1, y1 = int(roi[0] * width), int(roi[1] * height)
        x2, y2 = int(roi[2] * width), int(roi[3] * height)
        color = (60, 220, 60) if index == highlight else (255, 180, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1 + 6, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


def regions_from_config(config_path: Path) -> list[tuple[str, tuple[float, float, float, float]]]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    regions: list[tuple[str, tuple[float, float, float, float]]] = []
    for item in raw.get("desks", []):
        roi = tuple(float(v) for v in item.get("roi", [0, 0, 1, 1]))
        regions.append((str(item.get("label", "Kursi")), roi))  # type: ignore[arg-type]
    return regions


def save_to_config(config_path: Path, regions) -> None:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    backup = config_path.with_suffix(".yaml.bak")
    backup.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    raw["desks"] = [
        {"id": index + 1, "label": label, "employee_id": None, "roi": [round(v, 4) for v in roi]}
        for index, (label, roi) in enumerate(regions)
    ]
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    LOGGER.info("config.yaml diperbarui (backup: %s)", backup.name)


def save_to_db(cfg, regions) -> None:
    camera_id = db.ensure_camera(cfg.camera.id, cfg.camera.name, str(cfg.camera.source))
    db.sync_desks(
        camera_id,
        [{"label": label, "roi": roi, "employee_id": None} for label, roi in regions],
    )
    LOGGER.info("%d area kursi disimpan ke database", len(regions))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kalibrasi area (ROI) kursi kerja")
    parser.add_argument("--config", default=None)
    parser.add_argument("--source", default=None, help="0/1 untuk kamera, path video untuk uji")
    parser.add_argument("--image", default=None, help="pakai satu foto statis")
    parser.add_argument("--no-config-save", action="store_true")
    parser.add_argument("--no-db-save", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config(args.config)
    config_path = PROJECT_ROOT / (args.config or "config.yaml")
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    regions = regions_from_config(config_path)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            LOGGER.error("gagal membaca gambar %s", args.image)
            return 2
        capture = None
    else:
        source = parse_source(args.source, cfg.camera.source)
        capture = grab_frame(source)
        if capture is None:
            return 2
        ok, frame = capture.read()
        if not ok:
            LOGGER.error("tidak bisa membaca frame")
            return 2

    LOGGER.info("a= tambah area, d= hapus area terakhir, s= simpan, w= simpan&keluar, q= keluar")
    try:
        while True:
            cv2.imshow("Calibrate ROI", draw(frame.copy(), regions))
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("a"):
                box = cv2.selectROI("Pilih area kursi", frame, showCrosshair=True, fromCenter=False)
                if box == (0, 0, 0, 0):
                    LOGGER.info("dibatalkan")
                    continue
                x, y, w, h = box
                height, width = frame.shape[:2]
                roi = (x / width, y / height, (x + w) / width, (y + h) / height)
                label = input(f"label area #{len(regions) + 1} (mis. Kursi 1): ").strip() or f"Kursi {len(regions) + 1}"
                regions.append((label, roi))
                LOGGER.info("ditambah %s -> %s", label, [round(v, 4) for v in roi])
            elif key == ord("d"):
                if regions:
                    LOGGER.info("dihapus %s", regions.pop()[0])
            elif key == ord("s"):
                if not args.no_db_save:
                    save_to_db(cfg, regions)
                if not args.no_config_save:
                    save_to_config(config_path, regions)
            elif key == ord("w"):
                if not args.no_db_save:
                    save_to_db(cfg, regions)
                if not args.no_config_save:
                    save_to_config(config_path, regions)
                break
    finally:
        cv2.destroyAllWindows()
        if capture is not None:
            capture.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())