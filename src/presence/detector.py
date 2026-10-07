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

    @property
    def upper_center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y1 + 0.35 * max(1, y2 - y1))


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
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
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
            half=(self._device != "cpu"),
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
    anchor: tuple[float, float] | None = None,
) -> list[tuple[int, float, bool]]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = detection.bbox
    area = max(1, (x2 - x1) * (y2 - y1))
    center_x, center_y = detection.center

    # Anchor prioritizes torso/head if available, otherwise upper body center
    if anchor is not None and anchor != (0.0, 0.0):
        anchor_x, anchor_y = anchor
    else:
        anchor_x, anchor_y = detection.upper_center

    candidates: list[tuple[int, float, bool]] = []
    for desk_id, _label, roi in desks:
        rx1, ry1 = roi[0] * width, roi[1] * height
        rx2, ry2 = roi[2] * width, roi[3] * height

        inter_w = max(0.0, min(x2, rx2) - max(x1, rx1))
        inter_h = max(0.0, min(y2, ry2) - max(y1, ry1))
        inter = inter_w * inter_h

        anchor_inside = rx1 <= anchor_x <= rx2 and ry1 <= anchor_y <= ry2
        center_inside = rx1 <= center_x <= rx2 and ry1 <= center_y <= ry2

        if inter <= 0:
            if anchor_inside or center_inside:
                candidates.append((desk_id, 2.5 if anchor_inside else 0.5, anchor_inside or center_inside))
            continue

        ratio_person = inter / area
        desk_area = max(1.0, (rx2 - rx1) * (ry2 - ry1))
        ratio_desk = inter / desk_area

        # A desk is a candidate if:
        # 1. Anchor (torso/head) is inside the desk, OR
        # 2. Bbox center is inside the desk, OR
        # 3. Person's body overlap is substantial (>= min_ratio), OR
        # 4. Desk is covered (>= min_ratio) AND person is at least moderately inside (>= 0.25)
        is_candidate = (
            anchor_inside
            or center_inside
            or ratio_person >= min_ratio
            or (ratio_desk >= min_ratio and ratio_person >= 0.25)
        )
        if not is_candidate:
            continue

        # Score gives strong weight to anchor_inside (the chair where body/head is seated)
        score = (
            3.0 * ratio_person
            + (3.5 if anchor_inside else 0.0)
            + (1.0 if center_inside else 0.0)
            + 0.5 * min(1.0, ratio_desk)
        )
        candidates.append((desk_id, score, anchor_inside or center_inside))
    return candidates


def assign_to_desks(
    detections: list[Detection],
    desks: list[tuple[int, str, tuple[float, float, float, float]]],
    frame_shape: tuple[int, int],
    min_ratio: float = 0.5,
    anchors: dict[int, tuple[float, float]] | None = None,
) -> dict[int, Detection]:
    per_desk: dict[int, tuple[float, Detection]] = {}
    for detection in detections:
        anchor = anchors.get(detection.track_id) if anchors else None
        candidates = candidate_desks(detection, desks, frame_shape, min_ratio, anchor=anchor)
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
        self._pending_switch: dict[int, tuple[int, float]] = {}

    def assign(
        self,
        detections: list[Detection],
        desks: list[tuple[int, str, tuple[float, float, float, float]]],
        frame_shape: tuple[int, int],
        min_ratio: float,
        now: float,
        anchors: dict[int, tuple[float, float]] | None = None,
    ) -> dict[int, Detection]:
        assigned_candidates: list[tuple[int, float, Detection]] = []
        seen: set[int] = set()

        for detection in detections:
            track_id = detection.track_id
            if track_id < 0:
                continue
            seen.add(track_id)

            anchor = anchors.get(track_id) if anchors else None
            candidates = candidate_desks(detection, desks, frame_shape, min_ratio, anchor=anchor)

            if not candidates:
                previous = self._assignment.get(track_id)
                if previous is not None:
                    prev_desk, last_seen = previous
                    if now - last_seen < self.switch_grace_sec:
                        assigned_candidates.append((prev_desk, 0.5, detection))
                    else:
                        self._assignment.pop(track_id, None)
                        self._pending_switch.pop(track_id, None)
                continue

            best_desk, best_score, best_anchor_inside = max(candidates, key=lambda item: item[1])

            previous = self._assignment.get(track_id)
            if previous is None:
                self._assignment[track_id] = (best_desk, now)
                self._pending_switch.pop(track_id, None)
                assigned_candidates.append((best_desk, best_score, detection))
                continue

            prev_desk, last_seen = previous
            if best_desk == prev_desk:
                self._assignment[track_id] = (prev_desk, now)
                self._pending_switch.pop(track_id, None)
                assigned_candidates.append((prev_desk, best_score, detection))
                continue

            # best_desk != prev_desk: check if previous desk is still a valid candidate
            prev_match = next((c for c in candidates if c[0] == prev_desk), None)
            if prev_match is None:
                # prev_desk has zero candidate overlap now
                if now - last_seen >= self.switch_grace_sec:
                    self._assignment[track_id] = (best_desk, now)
                    self._pending_switch.pop(track_id, None)
                    assigned_candidates.append((best_desk, best_score, detection))
                else:
                    assigned_candidates.append((prev_desk, 0.5, detection))
                continue

            # Both prev_desk and best_desk are candidates
            prev_score, prev_anchor_inside = prev_match[1], prev_match[2]

            # Clear switch condition:
            # 1. best_desk contains anchor while prev_desk does not, OR
            # 2. best_score is substantially higher than prev_score
            should_switch = (
                (best_anchor_inside and not prev_anchor_inside)
                or (best_score >= prev_score * 1.35 + 0.5)
            )

            if should_switch:
                pending = self._pending_switch.get(track_id)
                confirm_time = 0.8 if best_anchor_inside else min(2.0, self.switch_grace_sec / 2)
                if pending is not None and pending[0] == best_desk:
                    if (now - pending[1]) >= confirm_time:
                        self._assignment[track_id] = (best_desk, now)
                        self._pending_switch.pop(track_id, None)
                        assigned_candidates.append((best_desk, best_score, detection))
                        continue
                else:
                    self._pending_switch[track_id] = (best_desk, now)

            # Keep previous desk while switch is pending or not meeting threshold
            self._assignment[track_id] = (prev_desk, now)
            assigned_candidates.append((prev_desk, prev_score, detection))

        for stale in [t for t in self._assignment if t not in seen]:
            self._assignment.pop(stale, None)
            self._pending_switch.pop(stale, None)

        result: dict[int, tuple[float, Detection]] = {}
        for desk_id, score, det in assigned_candidates:
            if desk_id not in result or score > result[desk_id][0]:
                result[desk_id] = (score, det)

        return {desk_id: det for desk_id, (_s, det) in result.items()}

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