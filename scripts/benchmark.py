from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from presence.config import load_config  # noqa: E402
from presence.detector import PersonDetector, face_region  # noqa: E402
from presence.face_id import build_identifier  # noqa: E402


def bench(name: str, fn, frames: int) -> float:
    fn()
    start = time.perf_counter()
    for _ in range(frames):
        fn()
    elapsed = (time.perf_counter() - start) / frames * 1000
    print(f"  {name:34s} {elapsed:7.1f} ms  ({1000 / elapsed:5.1f} fps)")
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--source", default="0")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"torch {torch.__version__} threads={torch.get_num_threads()} cuda={torch.cuda.is_available()}")

    source = int(args.source) if str(args.source).isdigit() else args.source
    capture = cv2.VideoCapture(source)
    frames = []
    while len(frames) < args.frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        print("tidak ada frame dari sumber")
        return 2
    frame = frames[-1]
    print(f"frame {frame.shape[1]}x{frame.shape[0]}, {len(frames)} frame diuji")

    if args.threads:
        torch.set_num_threads(args.threads)
        print(f"torch threads -> {torch.get_num_threads()}")

    identifier = build_identifier(cfg.face, [], backend="sface")

    for imgsz in (640, 512, 448, 384, 320):
        cfg.detection.imgsz = imgsz
        detector = PersonDetector(cfg.detection, PROJECT_ROOT)
        total = 0.0
        for sample in frames:
            start = time.perf_counter()
            detector.track(sample)
            total += time.perf_counter() - start
        ms = total / len(frames) * 1000
        print(f"  yolov8n imgsz={imgsz:<4d}{'':13s} {ms:7.1f} ms  ({1000 / ms:5.1f} fps)")

    cfg.detection.imgsz = 640
    detector = PersonDetector(cfg.detection, PROJECT_ROOT)
    result = detector.track(frame)
    det = result.detections[0] if result.detections else None
    if det is not None:
        for head_ratio in (0.55, 0.7, 0.85, 1.0):
            crop = face_region(frame, det.bbox, head_ratio, cfg.face.region_top_offset, cfg.face.widen, cfg.face.min_face_px)
            faces = identifier.detect(crop, min_face_px=10) if crop is not None else []
            tag = "crop None" if crop is None else f"crop {crop.shape[1]}x{crop.shape[0]}"
            print(f"  head_ratio={head_ratio}: {tag} wajah={len(faces)}")

    if det is not None:
        crop = head_crop(frame, det.bbox, 0.6, cfg.face.min_face_px)
        if crop is not None:
            bench("yunet+sface (match)", lambda: identifier.match(crop), len(frames))
            bench("yunet saja (detect)", lambda: identifier.detect(crop, min_face_px=10), len(frames))
    bench("yolov8n imgsz=640 (1x)", lambda: detector.track(frame), len(frames))
    return 0


if __name__ == "__main__":
    sys.exit(main())