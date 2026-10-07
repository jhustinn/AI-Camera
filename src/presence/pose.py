from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import PoseConfig
from .detector import resolve_device

LOGGER = logging.getLogger(__name__)

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

BONES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

BONE_COLORS = [
    (255, 180, 0),
    (255, 180, 0),
    (255, 220, 120),
    (255, 220, 120),
    (60, 200, 255),
    (60, 200, 255),
    (80, 230, 120),
    (80, 230, 120),
    (80, 230, 120),
    (80, 230, 120),
    (255, 120, 120),
    (200, 120, 255),
    (200, 120, 255),
    (255, 120, 120),
    (255, 120, 120),
    (255, 120, 120),
    (255, 120, 120),
    (255, 120, 120),
]

SITTING = "DUDUK"
STANDING = "BERDIRI"
UNKNOWN = "-"


@dataclass
class Skeleton:
    keypoints: np.ndarray
    score: float

    def center(self) -> tuple[float, float]:
        valid = self.keypoints[self.keypoints[:, 2] > 0.2]
        if len(valid) == 0:
            return (0.0, 0.0)
        return (float(valid[:, 0].mean()), float(valid[:, 1].mean()))

    def valid_count(self, min_conf: float) -> int:
        return int((self.keypoints[:, 2] >= min_conf).sum())

    def posture(
        self,
        min_conf: float = 0.3,
        sitting_knee_ratio: float = 0.45,
        standing_knee_ratio: float = 0.6,
        min_keypoints: int = 8,
    ) -> str:
        if self.valid_count(min_conf) < min_keypoints:
            return UNKNOWN
        return classify_posture(
            self.keypoints,
            min_conf=min_conf,
            sitting_knee_ratio=sitting_knee_ratio,
            standing_knee_ratio=standing_knee_ratio,
        )


def _angle_degrees(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    va = a[:2] - vertex[:2]
    vb = b[:2] - vertex[:2]
    norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if norm < 1e-6:
        return float("nan")
    cosine = float(np.dot(va, vb) / norm)
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cosine)))))


def clip_to_bbox(
    skeleton: Skeleton,
    bbox: tuple[int, int, int, int],
    margin: float = 0.05,
) -> tuple[Skeleton, float]:
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * margin
    pad_y = (y2 - y1) * margin
    min_x, max_x = x1 - pad_x, x2 + pad_x
    min_y, max_y = y1 - pad_y, y2 + pad_y
    points = skeleton.keypoints.copy()
    inside = (
        (points[:, 0] >= min_x)
        & (points[:, 0] <= max_x)
        & (points[:, 1] >= min_y)
        & (points[:, 1] <= max_y)
        & (points[:, 2] > 0)
    )
    valid_before = points[points[:, 2] > 0]
    ratio = float(inside.sum() / len(valid_before)) if len(valid_before) else 0.0
    points[:, 2] = np.where(inside, points[:, 2], 0.0).astype(np.float32)
    return Skeleton(keypoints=points, score=skeleton.score), ratio


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def validate_skeleton(
    skeleton: Skeleton,
    min_conf: float = 0.3,
    limb_ratio_range: tuple[float, float] = (0.3, 1.8),
    shoulder_ratio_max: float = 2.5,
) -> Skeleton:
    points = skeleton.keypoints.copy()

    def valid(index: int) -> bool:
        return points[index][2] >= min_conf

    shoulders_ok = valid(5) and valid(6)
    hips_ok = valid(11) and valid(12)
    torso: float | None = None
    if shoulders_ok and hips_ok:
        shoulder_mid = (points[5][:2] + points[6][:2]) / 2.0
        hip_mid = (points[11][:2] + points[12][:2]) / 2.0
        torso = float(np.linalg.norm(shoulder_mid - hip_mid))
        if torso < 1e-3:
            torso = None

    if shoulders_ok and torso:
        shoulder_span = _distance(points[5], points[6])
        if shoulder_span > shoulder_ratio_max * torso:
            points[6][2] = 0.0

    low, high = limb_ratio_range
    for hip_index, knee_index, ankle_index in ((11, 13, 15), (12, 14, 16)):
        if not (valid(hip_index) and valid(knee_index)):
            continue
        if torso:
            upper = _distance(points[hip_index], points[knee_index])
            if upper < low * torso or upper > high * torso:
                points[knee_index][2] = 0.0
                if valid(ankle_index):
                    points[ankle_index][2] = 0.0
                continue
        if valid(ankle_index):
            lower = _distance(points[knee_index], points[ankle_index])
            if torso:
                if lower < low * torso or lower > high * torso:
                    points[ankle_index][2] = 0.0
                    continue
            if lower > 0 and _distance(points[hip_index], points[knee_index]) > 0:
                if lower / max(_distance(points[hip_index], points[knee_index]), 1e-6) > 3.0:
                    points[ankle_index][2] = 0.0

    return Skeleton(keypoints=points, score=skeleton.score)


def smooth_skeleton(current: Skeleton, previous: Skeleton | None, alpha: float) -> Skeleton:
    alpha = min(1.0, max(0.0, alpha))
    points = current.keypoints.copy()
    if previous is not None and previous.keypoints.shape == points.shape and alpha < 1.0:
        both = (points[:, 2] > 0) & (previous.keypoints[:, 2] > 0)
        if both.any():
            blended = alpha * points[both, :2] + (1.0 - alpha) * previous.keypoints[both, :2]
            points[both, :2] = blended.astype(np.float32)
    return Skeleton(keypoints=points, score=current.score)


KNEE_SITTING_ANGLE = 120.0
KNEE_STANDING_ANGLE = 150.0


def classify_posture(
    keypoints: np.ndarray,
    min_conf: float = 0.3,
    sitting_knee_ratio: float = 0.45,
    standing_knee_ratio: float = 0.6,
) -> str:
    def valid(index: int) -> np.ndarray | None:
        point = keypoints[index]
        return None if point[2] < min_conf else point

    shoulders = [valid(5), valid(6)]
    hips = [valid(11), valid(12)]
    if any(p is None for p in shoulders) or any(p is None for p in hips):
        return UNKNOWN
    shoulder_mid = np.mean([p[:2] for p in shoulders if p is not None], axis=0)
    hip_mid = np.mean([p[:2] for p in hips if p is not None], axis=0)
    torso = abs(float(shoulder_mid[1] - hip_mid[1]))
    if torso < 1e-3:
        torso = float(np.linalg.norm(shoulder_mid - hip_mid))
    if torso < 1e-3:
        return UNKNOWN

    ankles = [valid(15), valid(16)]
    knees = [valid(13), valid(14)]
    knee_angles = [
        _angle_degrees(hips[i], knees[i], ankles[i])
        for i in range(2)
        if hips[i] is not None and knees[i] is not None and ankles[i] is not None
    ]
    knee_angles = [a for a in knee_angles if not np.isnan(a)]
    knee_angle = float(np.median(knee_angles)) if knee_angles else None

    knee_drop: float | None = None
    if all(p is not None for p in knees):
        knee_mid_y = float(np.mean([p[1] for p in knees if p is not None]))
        knee_drop = abs(knee_mid_y - float(hip_mid[1])) / torso

    if knee_angle is not None and knee_angle < KNEE_SITTING_ANGLE:
        return SITTING
    if knee_drop is not None and knee_drop < sitting_knee_ratio:
        return SITTING
    if knee_angle is not None and knee_angle > KNEE_STANDING_ANGLE:
        return STANDING
    if knee_drop is not None and knee_drop >= standing_knee_ratio:
        return STANDING
    return UNKNOWN


class PoseEstimator:
    def __init__(self, cfg: PoseConfig, project_root: Path | None = None) -> None:
        from ultralytics import YOLO

        self._cfg = cfg
        self._device = resolve_device(cfg.device)
        self._model = YOLO(cfg.model)
        self._project_root = project_root or Path(__file__).resolve().parents[2]

    @property
    def device(self) -> str:
        return self._device

    def estimate(self, frame: np.ndarray) -> list[Skeleton]:
        results = self._model.predict(
            frame,
            conf=self._cfg.confidence,
            imgsz=self._cfg.imgsz,
            device=self._device,
            verbose=False,
        )
        skeletons: list[Skeleton] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            keypoints = getattr(result, "keypoints", None)
            if boxes is None or keypoints is None or keypoints.data is None:
                continue
            if len(boxes) == 0:
                continue
            scores = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(boxes))
            for index in range(len(boxes)):
                points = keypoints.data[index].cpu().numpy().astype(np.float32)
                if points.shape[0] != len(KEYPOINT_NAMES):
                    continue
                skeletons.append(Skeleton(keypoints=points, score=float(scores[index])))
        return skeletons

    def draw(
        self,
        frame: np.ndarray,
        skeletons: list[Skeleton],
        min_conf: float | None = None,
        legs: bool | None = None,
    ) -> np.ndarray:
        cfg = self._cfg
        threshold = cfg.keypoint_confidence if min_conf is None else min_conf
        show_legs = cfg.legs if legs is None else legs
        for skeleton in skeletons:
            points = skeleton.keypoints
            for index, (start, end) in enumerate(BONES):
                if not show_legs and (start >= 11 or end >= 11):
                    continue
                a, b = points[start], points[end]
                if a[2] < threshold or b[2] < threshold:
                    continue
                color = BONE_COLORS[index] if index < len(BONE_COLORS) else (255, 255, 255)
                cv2.line(
                    frame,
                    (int(a[0]), int(a[1])),
                    (int(b[0]), int(b[1])),
                    color,
                    cfg.line_thickness,
                    cv2.LINE_AA,
                )
            for point_index, point in enumerate(points):
                if not show_legs and point_index >= 11:
                    continue
                if point[2] < threshold:
                    continue
                filled = int(min(255, 120 + point[2] * 135))
                cv2.circle(
                    frame,
                    (int(point[0]), int(point[1])),
                    cfg.radius,
                    (filled, filled, filled),
                    -1,
                    cv2.LINE_AA,
                )
        return frame

    def posture_for(self, skeleton: Skeleton, target: tuple[float, float]) -> str:
        return skeleton.posture(
            min_conf=self._cfg.keypoint_confidence,
            sitting_knee_ratio=self._cfg.sitting_knee_ratio,
            standing_knee_ratio=self._cfg.standing_knee_ratio,
            min_keypoints=self._cfg.min_keypoints,
        )


def match_skeletons(
    skeletons: list[Skeleton],
    detections: list[Any],
    min_conf: float = 0.3,
    max_distance: float = 400.0,
    min_keypoints: int = 4,
) -> dict[int, Skeleton]:
    result: dict[int, Skeleton] = {}
    used: set[int] = set()
    for detection in detections:
        if detection.track_id < 0 or detection.track_id in used:
            continue
        best_index = -1
        best_distance = float("inf")
        for index, skeleton in enumerate(skeletons):
            if index in used or skeleton.valid_count(min_conf) < min_keypoints:
                continue
            distance = float(np.linalg.norm(np.array(skeleton.center()) - np.array(detection.center)))
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index >= 0 and best_distance < max_distance:
            used.add(best_index)
            result[detection.track_id] = skeletons[best_index]
    return result