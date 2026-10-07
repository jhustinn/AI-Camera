from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import DetectionConfig

LOGGER = logging.getLogger(__name__)

TRACKER_TEMPLATE = """tracker_type: bytetrack
track_buffer: {track_buffer}
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
match_thresh: 0.8
fuse_score: true
"""


@dataclass
class Detection:
    track_id: int
    bbox: tuple[int, int, int, int]
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class DetectionResult:
    detections: list[Detection]
    fps: float
    frame_id: int


def resolve_device(requested: str) -> str:
    import torch

    if not requested or requested == "cpu":
        return "cpu"
    wants_cuda = requested.startswith("cuda") or requested.split(",")[0].strip().isdigit()
    if wants_cuda and not torch.cuda.is_available():
        LOGGER.warning("CUDA tidak tersedia (torch %s), fallback ke CPU", torch.__version__)
        return "cpu"
    return requested


class PersonDetector:
    def __init__(self, cfg: DetectionConfig, project_root: Path | None = None) -> None:
        from ultralytics import YOLO

        self._cfg = cfg
        if cfg.torch_threads:
            import torch

            torch.set_num_threads(max(1, cfg.torch_threads))
        self._device = resolve_device(cfg.device)
        root = project_root or Path(__file__).resolve().parents[2]
        model_path = self._prepare_model(root)
        self._model = YOLO(str(model_path))
        tracker_cfg = root / "data" / f"tracker_{cfg.tracker}.yaml"
        tracker_cfg.parent.mkdir(parents=True, exist_ok=True)
        tracker_cfg.write_text(TRACKER_TEMPLATE.format(track_buffer=cfg.track_buffer), encoding="utf-8")
        self._tracker_cfg = tracker_cfg
        self._frame_id = 0

    def _prepare_model(self, root: Path) -> Path:
        name = self._cfg.model
        if not name.endswith(".onnx"):
            return Path(name)
        onnx_path = Path(name)
        if onnx_path.exists():
            return onnx_path
        pt_path = root / (name.replace(".onnx", ".pt"))
        LOGGER.info("export %s -> onnx (sekali saja)", pt_path.name)
        from ultralytics import YOLO

        YOLO(str(pt_path)).export(format="onnx", imgsz=self._cfg.imgsz, dynamic=False, simplify=False)
        return Path(name)

    @property
    def device(self) -> str:
        return self._device

    def track(self, frame: np.ndarray) -> DetectionResult:
        if self._cfg.max_width and frame.shape[1] > self._cfg.max_width:
            scale = self._cfg.max_width / frame.shape[1]
            frame = cv2.resize(
                frame,
                (self._cfg.max_width, max(1, int(frame.shape[0] * scale))),
                interpolation=cv2.INTER_LINEAR,
            )
        results = self._model.track(
            frame,
            persist=True,
            conf=self._cfg.confidence,
            iou=self._cfg.iou,
            imgsz=self._cfg.imgsz,
            classes=self._cfg.classes,
            device=self._device,
            tracker=str(self._tracker_cfg),
            verbose=False,
        )
        self._frame_id += 1
        detections: list[Detection] = []
        inference_ms = 0.0
        for result in results:
            speed = getattr(result, "speed", None)
            if isinstance(speed, dict):
                inference_ms = max(inference_ms, float(speed.get("inference", 0.0)))
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            ids = boxes.id
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for index in range(len(xyxy)):
                x1, y1, x2, y2 = [float(v) for v in xyxy[index]]
                track_id = int(ids[index].item()) if ids is not None else -1
                detections.append(
                    Detection(
                        track_id=track_id,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        conf=float(confs[index]),
                    )
                )
        fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
        return DetectionResult(detections=detections, fps=fps, frame_id=self._frame_id)


def face_region(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    region_ratio: float = 0.8,
    top_offset: float = 0.15,
    widen: float = 0.12,
    min_face_px: int = 40,
    max_width: int = 640,
) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    box_h = max(1, y2 - y1)
    box_w = max(1, x2 - x1)
    top = max(0, y1 - int(box_h * top_offset))
    bottom = min(height, top + max(1, int(box_h * region_ratio)))
    side = int(box_w * widen)
    left = max(0, x1 - side)
    right = min(width, x2 + side)
    if right - left < min_face_px or bottom - top < min_face_px:
        return None
    crop = frame[top:bottom, left:right]
    if max_width and crop.shape[1] > max_width:
        scale = max_width / crop.shape[1]
        crop = cv2.resize(
            crop,
            (max_width, max(1, int(crop.shape[0] * scale))),
            interpolation=cv2.INTER_LINEAR,
        )
    return crop


def candidate_desks(
    detection: Detection,
    desks: list[tuple[int, str, tuple[float, float, float, float]]],
    frame_shape: tuple[int, int],
    min_ratio: float = 0.5,
) -> list[tuple[int, float, bool]]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = detection.bbox
    area = max(1, (x2 - x1) * (y2 - y1))
    center_x, center_y = detection.center
    candidates: list[tuple[int, float, bool]] = []
    for desk_id, _label, roi in desks:
        rx1, ry1 = roi[0] * width, roi[1] * height
        rx2, ry2 = roi[2] * width, roi[3] * height
        inter_w = max(0.0, min(x2, rx2) - max(x1, rx1))
        inter_h = max(0.0, min(y2, ry2) - max(y1, ry1))
        inter = inter_w * inter_h
        center_inside = rx1 <= center_x <= rx2 and ry1 <= center_y <= ry2
        if inter <= 0:
            if center_inside:
                candidates.append((desk_id, 0.5, True))
            continue
        ratio_person = inter / area
        ratio_desk = inter / max(1.0, (rx2 - rx1) * (ry2 - ry1))
        if not (center_inside or ratio_person >= min_ratio or ratio_desk >= min_ratio):
            continue
        score = 2.0 * ratio_person + ratio_desk + (1.0 if center_inside else 0.0)
        candidates.append((desk_id, score, center_inside))
    return candidates


def assign_to_desks(
    detections: list[Detection],
    desks: list[tuple[int, str, tuple[float, float, float, float]]],
    frame_shape: tuple[int, int],
    min_ratio: float = 0.5,
) -> dict[int, Detection]:
    per_desk: dict[int, tuple[float, Detection]] = {}
    for detection in detections:
        candidates = candidate_desks(detection, desks, frame_shape, min_ratio)
        if not candidates:
            continue
        desk_id, score, _center = max(candidates, key=lambda item: item[1])
        if desk_id not in per_desk or score > per_desk[desk_id][0]:
            per_desk[desk_id] = (score, detection)
    return {desk_id: detection for desk_id, (_score, detection) in per_desk.items()}


class DeskAssigner:
    def __init__(self, switch_grace_sec: float = 8.0) -> None:
        self.switch_grace_sec = switch_grace_sec
        self._assignment: dict[int, tuple[int, float]] = {}

    def assign(
        self,
        detections: list[Detection],
        desks: list[tuple[int, str, tuple[float, float, float, float]]],
        frame_shape: tuple[int, int],
        min_ratio: float,
        now: float,
    ) -> dict[int, Detection]:
        result: dict[int, Detection] = {}
        seen: set[int] = set()
        for detection in detections:
            track_id = detection.track_id
            if track_id < 0:
                continue
            seen.add(track_id)
            candidates = candidate_desks(detection, desks, frame_shape, min_ratio)
            candidate_ids = {desk_id for desk_id, _score, _center in candidates}
            previous = self._assignment.get(track_id)
            if previous is not None:
                prev_desk, last_seen = previous
                if prev_desk in candidate_ids:
                    self._assignment[track_id] = (prev_desk, now)
                    result[prev_desk] = detection
                    continue
                if now - last_seen < self.switch_grace_sec:
                    result[prev_desk] = detection
                    continue
            if candidates:
                desk_id = max(candidates, key=lambda item: item[1])[0]
                self._assignment[track_id] = (desk_id, now)
                result[desk_id] = detection
            else:
                self._assignment.pop(track_id, None)
        for stale in [t for t in self._assignment if t not in seen]:
            self._assignment.pop(stale, None)
        return result

    def desk_of(self, track_id: int) -> int | None:
        entry = self._assignment.get(track_id)
        return entry[0] if entry else None


def draw_rois(frame: np.ndarray, desks: list[tuple[int, str, tuple[float, float, float, float]]]) -> np.ndarray:
    height, width = frame.shape[:2]
    for _desk_id, label, roi in desks:
        x1, y1 = int(roi[0] * width), int(roi[1] * height)
        x2, y2 = int(roi[2] * width), int(roi[3] * height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(frame, label, (x1 + 6, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)
    return frame