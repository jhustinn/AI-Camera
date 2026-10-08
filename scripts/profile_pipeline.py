from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse  # noqa: E402
import glob  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from presence import db  # noqa: E402
from presence.config import load_config  # noqa: E402
from presence.detector import PersonDetector, assign_to_desks, face_region  # noqa: E402
from presence.face_id import build_identifier  # noqa: E402
from presence.pose import PoseEstimator, clip_to_bbox, smooth_skeleton, validate_skeleton  # noqa: E402


def timeit(fn, frames, warmup=2):
    for _ in range(warmup):
        fn(frames[0])
    start = time.perf_counter()
    for frame in frames:
        fn(frame)
    return 1000 * (time.perf_counter() - start) / len(frames)


def main() -> int:
    parser = argparse.ArgumentParser(description="Profil biaya per tahap pipeline")
    parser.add_argument("--frames", default="data/seq")
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    folder = Path(args.frames)
    if not folder.is_absolute():
        folder = PROJECT_ROOT / folder
    paths = sorted(glob.glob(str(folder / "*.jpg")))[: args.count]
    frames = [cv2.imread(p) for p in paths]
    frames = [f for f in frames if f is not None]
    if not frames:
        print("tidak ada frame untuk diprofil")
        return 2
    print(f"frame: {frames[0].shape[1]}x{frames[0].shape[0]}, jumlah {len(frames)}\n")

    detector = PersonDetector(cfg.detection, PROJECT_ROOT)
    pose = PoseEstimator(cfg.pose, PROJECT_ROOT)
    identifier = build_identifier(cfg.face, db.load_known_faces(), backend="sface")
    desks = [(d.id, d.label, d.roi) for d in cfg.desks]

    detect_ms = timeit(lambda f: detector.track(f), frames)
    pose_ms = timeit(lambda f: pose.estimate(f), frames)
    result = detector.track(frames[-1])
    dets = result.detections

    face_ms = 0.0
    if dets:
        def do_face(frame):
            detection = dets[0]
            crop = face_region(
                frame, detection.bbox, cfg.face.region_ratio, cfg.face.region_top_offset,
                cfg.face.widen, cfg.face.min_face_px, cfg.face.crop_max_width,
            )
            if crop is not None:
                identifier.match(crop)

        face_ms = timeit(do_face, frames)

    overlay_ms = timeit(
        lambda f: cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, cfg.server.jpeg_quality]), frames
    )
    desk_ms = timeit(lambda f: assign_to_desks(dets, desks, f.shape, cfg.presence.desk_overlap_ratio), frames)

    db_ms = 0.0
    try:
        payload = {
            "desk_id": 1, "label": "Kursi 1", "status": "PRESENT", "employee_id": None,
            "employee_name": None, "track_id": 1, "sit_since": None, "duration_sec": 1,
            "away_sec": 0, "session_id": None, "fps": 30.0,
        }
        start = time.perf_counter()
        for _ in range(5):
            db.upsert_live_status(payload)
        db_ms = 1000 * (time.perf_counter() - start) / 5
    except Exception as exc:  # noqa: BLE001
        print(f"(DB tidak diuji: {exc})")

    face_every = max(1, cfg.face.run_every_n_frames)
    pose_every = max(1, cfg.pose.every_n_frames)
    amortized_face = face_ms / face_every
    amortized_pose = pose_ms / pose_every
    total = detect_ms + amortized_face + amortized_pose + overlay_ms + desk_ms
    print(f"  deteksi orang (YOLOv8n) : {detect_ms:7.2f} ms   tiap frame")
    print(f"  stickman (YOLOv8n-pose)  : {pose_ms:7.2f} ms   tiap {pose_every} frame -> {amortized_pose:6.2f} ms/frame")
    print(f"  wajah (YuNet+SFace)      : {face_ms:7.2f} ms   tiap {face_every} frame -> {amortized_face:6.2f} ms/frame")
    print(f"  assignasi ROI            : {desk_ms:7.2f} ms")
    print(f"  encode JPEG (stream)     : {overlay_ms:7.2f} ms   tiap frame")
    print(f"  tulis DB live_status     : {db_ms:7.2f} ms   per_desk per {cfg.server.poll_seconds}s")
    print(f"\n  TOTAL per frame (amortisasi) : {total:6.2f} ms  -> {1000 / total:5.1f} fps teoretis")
    print(f" acity: 1 frame = {frames[0].shape[1]}x{frames[0].shape[0]}, imgsz deteksi {cfg.detection.imgsz}, imgsz pose {cfg.pose.imgsz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())