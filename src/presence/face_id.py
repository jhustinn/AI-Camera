from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import FaceConfig

LOGGER = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

WEIGHTS = {
    "yunet": (
        "face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ),
    "sface": (
        "face_recognition_sface_2021dec.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ),
}

SFACE_THRESHOLD_DEFAULT = 0.363


def ensure_weights() -> dict[str, Path]:
    import urllib.request

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, (filename, url) in WEIGHTS.items():
        target = MODEL_DIR / filename
        if not target.exists() or target.stat().st_size == 0:
            LOGGER.info("downloading %s model", key)
            tmp = target.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(target)
        paths[key] = target
    return paths


@dataclass
class FaceMatch:
    employee_id: int | None
    employee_name: str | None
    similarity: float


class SFaceIdentifier:
    backend = "sface"

    def __init__(self, cfg: FaceConfig, known: list[dict[str, Any]] | None = None) -> None:
        self._cfg = cfg
        paths = ensure_weights()
        det_size = tuple(int(v) for v in cfg.det_size)
        self._detector = cv2.FaceDetectorYN.create(str(paths["yunet"]), "", det_size, 0.6, 0.3, 5000)
        self._detector.setScoreThreshold(float(cfg.score_threshold))
        self._detector.setNMSThreshold(0.3)
        self._recognizer = cv2.FaceRecognizerSF.create(str(paths["sface"]), "")
        self.set_known(known or [])

    def set_known(self, known: list[dict[str, Any]]) -> None:
        if known:
            matrix = np.stack([np.asarray(item["embedding"], dtype=np.float32) for item in known])
        else:
            matrix = np.zeros((0, 128), dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._matrix = matrix / np.clip(norms, 1e-8, None)
        self._ids = [int(item["id"]) for item in known]
        self._names = [str(item["name"]) for item in known]

    def detect(self, image: np.ndarray, min_face_px: int = 0) -> list[np.ndarray]:
        if image.size == 0:
            return []
        height, width = image.shape[:2]
        upscale = float(self._cfg.upscale)
        if upscale > 1.0:
            width = int(width * upscale)
            height = int(height * upscale)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        self._detector.setInputSize((width, height))
        try:
            _, faces = self._detector.detect(image)
        except cv2.error as exc:
            LOGGER.warning("yunet detect failed: %s", exc)
            return []
        if faces is None:
            return []
        min_face = min_face_px / max(1.0, upscale)
        crops: list[np.ndarray] = []
        for face in faces:
            x, y, fw, fh = [int(v) for v in face[:4]]
            if min_face and (fw < min_face or fh < min_face):
                continue
            if x < 0 or y < 0 or x + fw >= width or y + fh >= height:
                pad = np.zeros((fh, fw, 3), dtype=np.uint8)
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(width, x + fw), min(height, y + fh)
                if x1 <= x0 or y1 <= y0:
                    continue
                pad[: y1 - y0, : x1 - x0] = image[y0:y1, x0:x1]
                crops.append(pad)
            else:
                crops.append(image[y : y + fh, x : x + fw])
        return crops

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        vector = self._recognizer.feature(aligned_face)
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        return vector / max(float(np.linalg.norm(vector)), 1e-8)

    def embed_image(self, image: np.ndarray, min_face_px: int = 0) -> np.ndarray | None:
        crops = self.detect(image, min_face_px=min_face_px)
        if not crops:
            return None
        biggest = max(crops, key=lambda c: c.shape[0] * c.shape[1])
        if biggest.shape[0] < 32:
            return None
        aligned = cv2.resize(biggest, (112, 112), interpolation=cv2.INTER_LINEAR)
        return self.embed(aligned)

    def match(self, image: np.ndarray) -> FaceMatch:
        crops = self.detect(image, min_face_px=self._cfg.min_face_px)
        if not crops or self._matrix.shape[0] == 0:
            return FaceMatch(None, None, -1.0)
        best = FaceMatch(None, None, -1.0)
        second = -1.0
        for crop in crops:
            aligned = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
            vector = self.embed(aligned)
            scores = self._matrix @ vector
            if scores.size == 1:
                order = [0]
            else:
                order = np.argsort(scores)[::-1]
            idx = int(order[0])
            similarity = float(scores[idx])
            runner_up = float(scores[int(order[1])]) if len(order) > 1 else -1.0
            if similarity > best.similarity:
                best = FaceMatch(self._ids[idx], self._names[idx], similarity)
                second = runner_up
        if best.similarity < self._cfg.similarity_threshold:
            return FaceMatch(None, best.employee_name, best.similarity)
        if self._cfg.min_margin > 0 and len(self._matrix) > 1:
            if (best.similarity - second) < self._cfg.min_margin:
                return FaceMatch(None, best.employee_name, best.similarity)
        return best


class InsightFaceIdentifier:
    backend = "insightface"

    def __init__(self, cfg: FaceConfig, known: list[dict[str, Any]] | None = None) -> None:
        from insightface.app import FaceAnalysis

        self._cfg = cfg
        self._app = FaceAnalysis(
            name=cfg.model, providers=list(cfg.providers), allowed_modules=["detection", "recognition"]
        )
        self._app.prepare(ctx_id=0, det_size=tuple(int(v) for v in cfg.det_size))
        self.set_known(known or [])

    def set_known(self, known: list[dict[str, Any]]) -> None:
        if known:
            matrix = np.stack([np.asarray(item["embedding"], dtype=np.float32) for item in known])
        else:
            matrix = np.zeros((0, 512), dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._matrix = matrix / np.clip(norms, 1e-8, None)
        self._ids = [int(item["id"]) for item in known]
        self._names = [str(item["name"]) for item in known]

    def embed_image(self, image: np.ndarray, min_face_px: int = 0) -> np.ndarray | None:
        faces = self._app.get(image)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        vector = np.asarray(face.normed_embedding, dtype=np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-8)

    def match(self, image: np.ndarray) -> FaceMatch:
        if self._matrix.shape[0] == 0:
            return FaceMatch(None, None, -1.0)
        faces = self._app.get(image)
        if not faces:
            return FaceMatch(None, None, -1.0)
        best = FaceMatch(None, None, -1.0)
        second = -1.0
        for face in faces:
            vector = np.asarray(face.normed_embedding, dtype=np.float32)
            scores = self._matrix @ vector
            order = [0] if scores.size == 1 else list(np.argsort(scores)[::-1])
            idx = int(order[0])
            similarity = float(scores[idx])
            runner_up = float(scores[int(order[1])]) if len(order) > 1 else -1.0
            if similarity > best.similarity:
                best = FaceMatch(self._ids[idx], self._names[idx], similarity)
                second = runner_up
        if best.similarity < self._cfg.similarity_threshold:
            return FaceMatch(None, best.employee_name, best.similarity)
        if self._cfg.min_margin > 0 and len(self._matrix) > 1:
            if (best.similarity - second) < self._cfg.min_margin:
                return FaceMatch(None, best.employee_name, best.similarity)
        return best


BACKENDS = {"sface": SFaceIdentifier, "insightface": InsightFaceIdentifier}


def build_identifier(cfg: FaceConfig, known: list[dict[str, Any]] | None = None, backend: str = "sface"):
    key = backend.lower()
    if key not in BACKENDS:
        raise ValueError(f"unknown face backend '{backend}', available: {sorted(BACKENDS)}")
    return BACKENDS[key](cfg, known)


class TrackIdentityRegistry:
    def __init__(self, votes_required: int = 4) -> None:
        self._votes_required = max(1, votes_required)
        self._votes: dict[int, dict[int, int]] = {}
        self._names: dict[int, dict[int, str]] = {}
        self._stable: dict[int, tuple[int, str | None]] = {}
        self._last: dict[int, tuple[int | None, str | None]] = {}
        self.seen_tracks: set[int] = set()

    def observe(self, track_id: int, match: FaceMatch) -> tuple[int | None, str | None, float]:
        self.seen_tracks.add(track_id)
        if match.employee_id is None:
            previous = self._last.get(track_id, (None, None))
            return previous[0], previous[1], match.similarity
        per_track = self._votes.setdefault(track_id, {})
        per_name = self._names.setdefault(track_id, {})
        per_track[match.employee_id] = per_track.get(match.employee_id, 0) + 1
        per_name[match.employee_id] = match.employee_name
        self._last[track_id] = (match.employee_id, match.employee_name)
        top_id, top_votes = max(per_track.items(), key=lambda kv: kv[1])
        if top_votes >= self._votes_required:
            self._stable[track_id] = (top_id, per_name.get(top_id))
        return self._stable.get(track_id, (match.employee_id, match.employee_name)) + (match.similarity,)

    def identity(self, track_id: int) -> tuple[int | None, str | None]:
        if track_id in self._stable:
            return self._stable[track_id]
        return self._last.get(track_id, (None, None))

    def release(self, track_id: int) -> tuple[int | None, str | None]:
        identity = self.identity(track_id)
        self._votes.pop(track_id, None)
        self._names.pop(track_id, None)
        self._stable.pop(track_id, None)
        self._last.pop(track_id, None)
        self.seen_tracks.discard(track_id)
        return identity

    def prune(self, active_track_ids: set[int]) -> None:
        for track_id in list(self.seen_tracks | set(self._votes)):
            if track_id not in active_track_ids:
                self.release(track_id)