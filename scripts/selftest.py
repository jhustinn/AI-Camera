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
from presence.detector import PersonDetector, assign_to_desks, head_crop  # noqa: E402
from presence.face_id import build_identifier  # noqa: E402

LOGGER = logging.getLogger("presence.selftest")
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"
SAMPLE_FILE = "bus.jpg"


def ensure_sample() -> Path:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    target = SAMPLES_DIR / SAMPLE_FILE
    if not target.exists() or target.stat().st_size == 0:
        LOGGER.info("mengunduh contoh gambar %s", SAMPLE_URL)
        urllib.request.urlretrieve(SAMPLE_URL, target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cek model deteksi & pengenalan wajah tanpa kamera/database")
    parser.add_argument("--image", default=None)
    parser.add_argument("--backend", default="sface", choices=["sface", "insightface"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config(args.config)
    if args.device:
        cfg.detection.device = args.device

    image_path = Path(args.image) if args.image else ensure_sample()
    frame = cv2.imread(str(image_path))
    if frame is None:
        LOGGER.error("gagal membaca gambar %s", image_path)
        return 2
    LOGGER.info("gambar %s ukuran %sx%s", image_path.name, frame.shape[1], frame.shape[0])

    failures: list[str] = []
    detector = PersonDetector(cfg.detection, PROJECT_ROOT)
    LOGGER.info("detector=%s device=%s", cfg.detection.model, detector.device)
    result = detector.track(frame)
    LOGGER.info("orang terdeteksi: %d (fps infer %.1f)", len(result.detections), result.fps)
    for detection in result.detections:
        LOGGER.info("  track_id=%s conf=%.2f bbox=%s", detection.track_id, detection.conf, detection.bbox)
    if len(result.detections) < 2:
        failures.append("deteksi orang kurang dari 2 (model ROI kelas person bermasalah?)")
    if result.fps <= 0:
        failures.append("tidak ada informasi kecepatan inferensi")

    desks = [(d.id, d.label, d.roi) for d in cfg.desks]
    assigned = assign_to_desks(result.detections, desks, frame.shape, cfg.presence.desk_overlap_ratio)
    LOGGER.info("pemetaan kursi: %s", {desk_id: det.track_id for desk_id, det in assigned.items()})

    identifier = build_identifier(cfg.face, [], backend=args.backend)
    LOGGER.info("face backend=%s", identifier.backend)
    vectors: list[np.ndarray] = []
    for detection in result.detections:
        crop = head_crop(frame, detection.bbox, cfg.face.head_ratio, min_face_px=20)
        if crop is None:
            LOGGER.warning("  crop kepala terlalu kecil untuk track_id=%s", detection.track_id)
            continue
        if hasattr(identifier, "detect"):
            faces = identifier.detect(crop, min_face_px=20)
            LOGGER.info("  track_id=%s wajah terdeteksi=%d", detection.track_id, len(faces))
            if not faces:
                continue
            aligned = cv2.resize(max(faces, key=lambda c: c.shape[0] * c.shape[1]), (112, 112))
        else:
            aligned = cv2.resize(crop, (112, 112))
        vectors.append(np.asarray(identifier.embed(aligned), dtype=np.float32))

    LOGGER.info("embedding wajah berhasil: %d", len(vectors))
    if len(vectors) < 2:
        failures.append("total embedding < 2, tidak bisa uji kemiripan")
    else:
        matrix = np.stack(vectors)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.clip(norms, 1e-8, None)
        scores = matrix @ matrix.T
        LOGGER.info("matriks kemiripan (cosine):\n%s", np.round(scores, 3))
        if float(np.max(np.diag(scores))) < 0.99:
            failures.append("embedding tidak ternormalisasi (bug normalisasi)")

    if failures:
        for failure in failures:
            LOGGER.error("GAGAL: %s", failure)
        return 1
    LOGGER.info("SELFTEST LULUS: deteksi orang, ROI, dan embedding wajah berfungsi")
    return 0


if __name__ == "__main__":
    sys.exit(main())