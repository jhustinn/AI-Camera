from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse  # noqa: E402
import logging  # noqa: E402
import urllib.request  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from presence.config import load_config  # noqa: E402
from presence.detector import PersonDetector  # noqa: E402
from presence.face_id import build_identifier  # noqa: E402

LOGGER = logging.getLogger("presence.demo")
SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"
DEMO_DIR = PROJECT_ROOT / "data" / "demo"
FACE_DIR = DEMO_DIR / "faces"


def ensure_sample() -> Path:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    target = DEMO_DIR / "bus.jpg"
    if not target.exists() or target.stat().st_size == 0:
        LOGGER.info("mengunduh %s", SAMPLE_URL)
        urllib.request.urlretrieve(SAMPLE_URL, target)
    return target


def extract_people(frame: np.ndarray, cfg, device: str) -> list[np.ndarray]:
    detector = PersonDetector(cfg.detection, PROJECT_ROOT)
    detector._device = detector._device if device is None else device
    result = detector.track(frame)
    people: list[np.ndarray] = []
    for detection in result.detections:
        x1, y1, x2, y2 = detection.bbox
        people.append(frame[max(0, y1) : y2, max(0, x1) : x2].copy())
    return people


def save_face_samples(people: list[np.ndarray], cfg, backend: str, min_size: int = 90) -> list[Path]:
    FACE_DIR.mkdir(parents=True, exist_ok=True)
    identifier = build_identifier(cfg.face, [], backend=backend)
    saved: list[Path] = []
    for index, person in enumerate(people, start=1):
        person_dir = FACE_DIR / f"orang_{index}"
        person_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for face in identifier.detect(person, min_face_px=min_size // 3):
            aligned = cv2.resize(face, (160, 160), interpolation=cv2.INTER_CUBIC)
            path = person_dir / f"face_{count + 1}.png"
            cv2.imwrite(str(path), aligned)
            saved.append(path)
            count += 1
            if count >= 2:
                break
    return saved


def paste(canvas: np.ndarray, person: np.ndarray, roi: tuple[float, float, float, float]) -> None:
    height, width = canvas.shape[:2]
    x1, y1 = int(roi[0] * width), int(roi[1] * height)
    x2, y2 = int(roi[2] * width), int(roi[3] * height)
    target_h = int((y2 - y1) * 1.1)
    scale = target_h / person.shape[0]
    target_w = max(1, int(person.shape[1] * scale))
    resized = cv2.resize(person, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    left = x1 + max(0, ((x2 - x1) - target_w) // 2)
    top = max(0, y2 - target_h - int((y2 - y1) * 0.02))
    canvas[top : top + target_h, left : left + target_w] = resized


def build_video(cfg, people: list[np.ndarray], out_path: Path, fps: int, width: int, height: int) -> dict:
    roi1 = tuple(cfg.desks[0].roi)
    roi2 = tuple(cfg.desks[1].roi)
    background = np.full((height, width, 3), 90, dtype=np.uint8)
    for roi in (roi1, roi2):
        x1, y1 = int(roi[0] * width), int(roi[1] * height)
        x2, y2 = int(roi[2] * width), int(roi[3] * height)
        cv2.rectangle(background, (x1, y1), (x2, y2), (150, 110, 60), -1)

    schedule = [
        (0, 20, None),
        (20, 80, (0, roi1)),
        (80, 95, None),
        (95, 155, (0, roi1)),
        (155, 200, None),
        (200, 280, (2, roi2)),
        (280, 320, None),
        (320, 360, (0, roi1)),
    ]

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit("gagal membuat file video")
    frames = 0
    try:
        for start, end, action in schedule:
            for _ in range(start, end):
                frame = background.copy()
                if action is not None:
                    index, roi = action
                    paste(frame, people[index], roi)
                writer.write(frame)
                frames += 1
    finally:
        writer.release()
    LOGGER.info("video demo ditulis: %s (%d frame @ %d fps)", out_path.name, frames, fps)
    return {"frames": frames, "seconds": frames / fps, "schedule": schedule}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Buat video demo untuk uji end-to-end tanpa kamera")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--backend", default="sface", choices=["sface", "insightface"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config(args.config)
    if args.device:
        cfg.detection.device = args.device

    sample = ensure_sample()
    frame = cv2.imread(str(sample))
    if frame is None:
        LOGGER.error("gagal membaca %s", sample)
        return 2
    people = extract_people(frame, cfg, args.device)
    LOGGER.info("orang diekstrak: %d", len(people))
    if len(people) < 2:
        LOGGER.error("butuh minimal 2 orang di gambar contoh")
        return 1

    faces = save_face_samples(people, cfg, args.backend)
    LOGGER.info("contoh wajah disimpan: %s", ", ".join(p.name for p in faces) or "-")

    out_path = Path(args.out) if args.out else DEMO_DIR / "demo.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    info = build_video(cfg, people, out_path, args.fps, args.width, args.height)
    LOGGER.info("timeline (detik): %s", [(s / args.fps, e / args.fps, a and a[0]) for s, e, a in info["schedule"]])
    print(f"video={out_path}")
    print(f"duration_seconds={info['seconds']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())