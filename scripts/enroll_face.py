from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from presence import db  # noqa: E402
from presence.config import load_config  # noqa: E402
from presence.face_id import build_identifier  # noqa: E402

LOGGER = logging.getLogger("presence.enroll")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_SAMPLES = 3


def average_embedding(vectors: list[np.ndarray]) -> np.ndarray:
    matrix = np.stack([np.asarray(v, dtype=np.float32) for v in vectors])
    mean = matrix.mean(axis=0)
    return mean / max(float(np.linalg.norm(mean)), 1e-8)


def enroll_from_folder(identifier, folder: Path, min_face_px: int) -> tuple[np.ndarray, int]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise SystemExit(f"tidak ada gambar di {folder}")
    vectors: list[np.ndarray] = []
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            LOGGER.warning("gagal membaca %s", path.name)
            continue
        vector = identifier.embed_image(image, min_face_px=min_face_px)
        if vector is None:
            LOGGER.warning("wajah tidak terdeteksi di %s", path.name)
            continue
        vectors.append(vector)
    return average_embedding(vectors), len(vectors)


def enroll_from_camera(identifier, source: Any, count: int, interval: float, min_face_px: int):
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"tidak bisa membuka kamera {source}")
    vectors: list[np.ndarray] = []
    last_capture = 0.0
    LOGGER.info("tekan 's' untuk ambil sampel manual, 'q' selesai (butuh minimal %d sampel)", MIN_SAMPLES)
    try:
        while len(vectors) < count:
            ok, frame = capture.read()
            if not ok:
                break
            display = frame.copy()
            manual = False
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                manual = True
                last_capture = 0.0
            now = time.monotonic()
            if now - last_capture >= interval or manual:
                vector = identifier.embed_image(frame, min_face_px=min_face_px)
                if vector is not None:
                    vectors.append(vector)
                    LOGGER.info("sampel %d/%d diambil", len(vectors), count)
                    last_capture = now
                elif manual:
                    LOGGER.info("wajah tidak terdeteksi, ulangi")
                    last_capture = now
            cv2.putText(
                display,
                f"sampel {len(vectors)}/{count}  [s] capture  [q] selesai",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Enroll Wajah", display)
    finally:
        cv2.destroyAllWindows()
        capture.release()
    if not vectors:
        raise SystemExit("tidak ada sampel wajah yang berhasil diambil")
    return average_embedding(vectors), len(vectors)


def enroll_from_video(identifier, video: str, cfg, count: int, start_frame: int = 0) -> tuple[np.ndarray, int]:
    from presence.detector import PersonDetector, face_region

    detector = PersonDetector(cfg.detection, PROJECT_ROOT)
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise SystemExit(f"tidak bisa membuka video {video}")
    vectors: list[np.ndarray] = []
    frame_id = 0
    try:
        while len(vectors) < count:
            ok, frame = capture.read()
            if not ok:
                break
            frame_id += 1
            if frame_id < start_frame:
                continue
            if frame_id % 5:
                continue
            for detection in detector.track(frame).detections:
                crop = face_region(
                    frame,
                    detection.bbox,
                    cfg.face.region_ratio,
                    cfg.face.region_top_offset,
                    cfg.face.widen,
                    cfg.face.min_face_px,
                    cfg.face.crop_max_width,
                )
                if crop is None:
                    continue
                faces = identifier.detect(crop, cfg.face.min_face_px)
                if not faces:
                    continue
                biggest = max(faces, key=lambda c: c.shape[0] * c.shape[1])
                vectors.append(identifier.embed(cv2.resize(biggest, (112, 112), interpolation=cv2.INTER_LINEAR)))
                LOGGER.info("sampel %d/%d diambil (frame %d)", len(vectors), count, frame_id)
                if len(vectors) >= count:
                    break
    finally:
        capture.release()
    if not vectors:
        raise SystemExit("tidak ada wajah yang terdeteksi di video")
    return average_embedding(vectors), len(vectors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Daftarkan wajah karyawan untuk face recognition")
    parser.add_argument("--name", required=True)
    parser.add_argument("--employee-no", default=None)
    parser.add_argument("--dept", default=None)
    parser.add_argument("--folder", default=None, help="folder berisi foto wajah")
    parser.add_argument("--video", default=None, help="ambil sampel dari file video (untuk uji)")
    parser.add_argument("--source", default=None, help="0/1 untuk kamera")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--start-frame", type=int, default=0, help="frame awal untuk sampling dari video")
    parser.add_argument("--interval", type=float, default=1.2)
    parser.add_argument("--backend", default="sface", choices=["sface", "insightface"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--min-face-px", type=int, default=16)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config(args.config)
    identifier = build_identifier(cfg.face, [], backend=args.backend)
    LOGGER.info("backend=%s, threshold=%.2f, upscale=%.1f", identifier.backend, cfg.face.similarity_threshold, cfg.face.upscale)

    if args.folder:
        embedding, samples = enroll_from_folder(identifier, Path(args.folder), args.min_face_px)
    elif args.video:
        embedding, samples = enroll_from_video(identifier, args.video, cfg, args.count, args.start_frame)
    else:
        source: Any = args.source
        if source is None:
            source = cfg.camera.source
        elif source.isdigit():
            source = int(source)
        embedding, samples = enroll_from_camera(identifier, source, args.count, args.interval, args.min_face_px)

    if samples < MIN_SAMPLES:
        LOGGER.warning("hanya %d sampel (minimal %d Disarankan), hasil mungkin tidak stabil", samples, MIN_SAMPLES)

    employee_id = db.add_employee(args.name, args.employee_no, args.dept, args.folder or None)
    db.save_employee_embedding(employee_id, embedding, samples)
    flag = PROJECT_ROOT / "data" / "reload_faces.flag"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.touch()

    LOGGER.info(
        "karyawan '%s' (id=%s) terdaftar dengan %d sampel wajah, dim=%d",
        args.name,
        employee_id,
        samples,
        int(embedding.shape[0]),
    )
    print(f"employee_id={employee_id} samples={samples} dim={embedding.shape[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())