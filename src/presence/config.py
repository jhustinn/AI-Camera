from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class CameraConfig:
    id: str
    name: str
    source: Any
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str = "MJPG"
    backend: str = "any"


@dataclass
class DetectionConfig:
    model: str = "yolov8n.pt"
    confidence: float = 0.45
    iou: float = 0.5
    device: str = "0"
    imgsz: int = 640
    unified: bool = False
    tracker: str = "bytetrack"
    track_buffer: int = 30
    max_width: int = 960
    classes: list[int] = field(default_factory=lambda: [0])
    torch_threads: int = 0


@dataclass
class FaceConfig:
    enabled: bool = True
    model: str = "buffalo_l"
    providers: list[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    det_size: tuple[int, int] = (320, 320)
    similarity_threshold: float = 0.4
    min_margin: float = 0.05
    run_every_n_frames: int = 5
    region_ratio: float = 0.8
    region_top_offset: float = 0.15
    widen: float = 0.12
    crop_max_width: int = 640
    upscale: float = 2.0
    score_threshold: float = 0.6
    min_face_px: int = 16
    votes_required: int = 3


@dataclass
class PresenceConfig:
    enter_confirm_sec: int = 10
    leave_confirm_sec: int = 10
    away_grace_sec: int = 60
    min_session_sec: int = 300
    desk_overlap_ratio: float = 0.5
    desk_switch_grace_sec: int = 8


@dataclass
class DeskConfig:
    id: int
    label: str
    roi: tuple[float, float, float, float]
    employee_id: int | None = None


@dataclass
class PoseConfig:
    enabled: bool = True
    model: str = "yolov8n-pose.pt"
    imgsz: int = 640
    confidence: float = 0.3
    keypoint_confidence: float = 0.3
    device: str = "0"
    every_n_frames: int = 2
    line_thickness: int = 2
    radius: int = 3
    posture_labels: bool = True
    min_keypoints: int = 8
    bbox_margin: float = 0.05
    smoothing: float = 0.25
    hold_frames: int = 10
    validate_limbs: bool = True
    legs: bool = False
    limb_ratio_min: float = 0.3
    limb_ratio_max: float = 1.8
    draw_min_keypoints: int = 5
    min_inbox_ratio: float = 0.6
    min_score: float = 0.5
    sitting_knee_ratio: float = 0.45
    standing_knee_ratio: float = 0.6


@dataclass
class CameraChannel:
    """Satu kamera CCTV dengan area kursi dan sumber videonya."""

    id: str
    name: str
    source: Any
    width: int = 1280
    height: int = 720
    fps: int = 25
    fourcc: str = "MJPG"
    backend: str = "dshow"
    stream_port: int = 8001
    desks: list[DeskConfig] = field(default_factory=list)


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    poll_seconds: int = 3
    stream_host: str = "127.0.0.1"
    stream_port: int = 8001
    stream_fps: int = 12
    jpeg_quality: int = 80


@dataclass
class LoggingConfig:
    level: str = "INFO"
    csv_fallback: bool = True
    screenshot_dir: str = "data/screenshots"


@dataclass
class AppConfig:
    timezone: str
    camera: CameraConfig
    detection: DetectionConfig
    face: FaceConfig
    presence: PresenceConfig
    desks: list[DeskConfig]
    pose: PoseConfig
    server: ServerConfig
    logging: LoggingConfig
    raw: dict[str, Any]
    channels: list[CameraChannel] = field(default_factory=list)

    @property
    def desk_by_id(self) -> dict[int, DeskConfig]:
        return {d.id: d for d in self.desks}

    def local_tz(self):
        from zoneinfo import ZoneInfo

        return ZoneInfo(self.timezone)


def _resolve_source(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


def _desk_list(items: Any, prefix: str = "") -> list[DeskConfig]:
    desks: list[DeskConfig] = []
    for index, item in enumerate(items or []):
        if not isinstance(item, dict):
            continue
        desks.append(
            DeskConfig(
                id=int(item.get("id", index + 1)),
                label=str(item.get("label", f"Kursi {index + 1}")),
                roi=tuple(float(v) for v in item.get("roi", [0, 0, 1, 1])),  # type: ignore[arg-type]
                employee_id=item.get("employee_id"),
            )
        )
    return desks


def _build_channels(
    raw: dict[str, Any],
    camera_raw: dict[str, Any],
    top_desks: list[DeskConfig],
    server_raw: dict[str, Any],
) -> list[CameraChannel]:
    """Daftar kamera aktif.

    Dua bentuk konfigurasi didukung:
    1. `cameras:` berisi beberapa entri (multi kamera CCTV).
    2. hanya `camera:` + `desks:` (satu kamera, bentuk lama).
    """
    entries = raw.get("cameras")
    base_port = int(server_raw.get("stream_port", 8001))
    if not entries:
        return [
            CameraChannel(
                id=str(camera_raw.get("id", "cam-1")),
                name=str(camera_raw.get("name", "Camera")),
                source=_resolve_source(camera_raw.get("source", 0)),
                width=int(camera_raw.get("width", 1280)),
                height=int(camera_raw.get("height", 720)),
                fps=int(camera_raw.get("fps", 25)),
                fourcc=str(camera_raw.get("fourcc", "MJPG")),
                backend=str(camera_raw.get("backend", "any")),
                stream_port=base_port,
                desks=top_desks,
            )
        ]

    channels: list[CameraChannel] = []
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            continue
        merged_desks = _desk_list(item.get("desks"), f"cam{index + 1}") or top_desks
        channels.append(
            CameraChannel(
                id=str(item.get("id", f"cam-{index + 1}")),
                name=str(item.get("name", f"Kamera {index + 1}")),
                source=_resolve_source(item.get("source", 0)),
                width=int(item.get("width", 1280)),
                height=int(item.get("height", 720)),
                fps=int(item.get("fps", 25)),
                fourcc=str(item.get("fourcc", "MJPG")),
                backend=str(item.get("backend", "any")),
                stream_port=int(item.get("stream_port", base_port + index)),
                desks=merged_desks,
            )
        )
    return channels


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(text: str) -> str:
    """Ganti ${VAR} dan ${VAR:-default} dengan nilai environment.

    Dipakai agar kredensial RTSP tidak perlu ditulis di dalam file config yang
    ikut ter-commit ke repository publik. Variabel yang tidak terisi akan
    dibiarkan apa adanya supaya salah konfigurasi kelihatan saat runtime.
    """
    def repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        value = os.environ.get(name)
        if value:
            return value
        return default if default is not None else match.group(0)

    return _ENV_PATTERN.sub(repl, text)


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path or os.getenv("PRESENCE_CONFIG", "config.yaml"))
    if not config_path.is_absolute():
        candidate = PROJECT_ROOT / config_path
        config_path = candidate if candidate.exists() else Path(config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(_expand_env(handle.read())) or {}

    camera_raw = raw.get("camera", {})
    detection_raw = raw.get("detection", {})
    face_raw = raw.get("face", {})
    presence_raw = raw.get("presence", {})
    pose_raw = raw.get("pose", {})
    server_raw = raw.get("server", {})
    logging_raw = raw.get("logging", {})

    desks = _desk_list(raw.get("desks"))

    det_size = face_raw.get("det_size", [320, 320])
    return AppConfig(
        timezone=str(raw.get("timezone", "Asia/Jakarta")),
        camera=CameraConfig(
            id=str(camera_raw.get("id", "cam-1")),
            name=str(camera_raw.get("name", "Camera")),
            source=_resolve_source(camera_raw.get("source", 0)),
            width=int(camera_raw.get("width", 1280)),
            height=int(camera_raw.get("height", 720)),
            fps=int(camera_raw.get("fps", 30)),
            fourcc=str(camera_raw.get("fourcc", "MJPG")),
            backend=str(camera_raw.get("backend", "any")),
        ),
        detection=DetectionConfig(
            model=str(detection_raw.get("model", "yolov8n.pt")),
            confidence=float(detection_raw.get("confidence", 0.45)),
            iou=float(detection_raw.get("iou", 0.5)),
            device=str(detection_raw.get("device", "0")),
            imgsz=int(detection_raw.get("imgsz", 480)),
            unified=bool(detection_raw.get("unified", False)),
            tracker=str(detection_raw.get("tracker", "bytetrack")),
            track_buffer=int(detection_raw.get("track_buffer", 30)),
            max_width=int(detection_raw.get("max_width", 960)),
            classes=[int(c) for c in detection_raw.get("classes", [0])],
            torch_threads=int(detection_raw.get("torch_threads", 0)),
        ),
        face=FaceConfig(
            enabled=bool(face_raw.get("enabled", True)),
            model=str(face_raw.get("model", "buffalo_l")),
            providers=list(face_raw.get("providers", ["CPUExecutionProvider"])),
            det_size=(int(det_size[0]), int(det_size[1])),
            similarity_threshold=float(face_raw.get("similarity_threshold", 0.4)),
            min_margin=float(face_raw.get("min_margin", 0.05)),
            run_every_n_frames=int(face_raw.get("run_every_n_frames", 5)),
            region_ratio=float(face_raw.get("region_ratio", 0.8)),
            region_top_offset=float(face_raw.get("region_top_offset", 0.15)),
            widen=float(face_raw.get("widen", 0.12)),
            crop_max_width=int(face_raw.get("crop_max_width", 640)),
            upscale=float(face_raw.get("upscale", 2.0)),
            score_threshold=float(face_raw.get("score_threshold", 0.6)),
            min_face_px=int(face_raw.get("min_face_px", 16)),
            votes_required=int(face_raw.get("votes_required", 3)),
        ),
        presence=PresenceConfig(
            enter_confirm_sec=int(presence_raw.get("enter_confirm_sec", 10)),
            leave_confirm_sec=int(presence_raw.get("leave_confirm_sec", 10)),
            away_grace_sec=int(presence_raw.get("away_grace_sec", 60)),
            min_session_sec=int(presence_raw.get("min_session_sec", 300)),
            desk_overlap_ratio=float(presence_raw.get("desk_overlap_ratio", 0.5)),
            desk_switch_grace_sec=int(presence_raw.get("desk_switch_grace_sec", 8)),
        ),
        desks=desks,
        pose=PoseConfig(
            enabled=bool(pose_raw.get("enabled", True)),
            model=str(pose_raw.get("model", "yolov8n-pose.pt")),
            imgsz=int(pose_raw.get("imgsz", 640)),
            confidence=float(pose_raw.get("confidence", 0.3)),
            keypoint_confidence=float(pose_raw.get("keypoint_confidence", 0.3)),
            device=str(pose_raw.get("device", "0")),
            every_n_frames=int(pose_raw.get("every_n_frames", 2)),
            line_thickness=int(pose_raw.get("line_thickness", 2)),
            radius=int(pose_raw.get("radius", 3)),
            posture_labels=bool(pose_raw.get("posture_labels", True)),
            min_keypoints=int(pose_raw.get("min_keypoints", 8)),
            bbox_margin=float(pose_raw.get("bbox_margin", 0.05)),
            smoothing=float(pose_raw.get("smoothing", 0.25)),
            hold_frames=int(pose_raw.get("hold_frames", 10)),
            validate_limbs=bool(pose_raw.get("validate_limbs", True)),
            legs=bool(pose_raw.get("legs", False)),
            draw_min_keypoints=int(pose_raw.get("draw_min_keypoints", 5)),
            min_inbox_ratio=float(pose_raw.get("min_inbox_ratio", 0.6)),
            min_score=float(pose_raw.get("min_score", 0.5)),
            limb_ratio_min=float(pose_raw.get("limb_ratio_min", 0.3)),
            limb_ratio_max=float(pose_raw.get("limb_ratio_max", 1.8)),
            sitting_knee_ratio=float(pose_raw.get("sitting_knee_ratio", 0.45)),
            standing_knee_ratio=float(pose_raw.get("standing_knee_ratio", 0.6)),
        ),
        server=ServerConfig(
            host=str(server_raw.get("host", "127.0.0.1")),
            port=int(server_raw.get("port", 8000)),
            poll_seconds=int(server_raw.get("poll_seconds", 3)),
            stream_host=str(server_raw.get("stream_host", "127.0.0.1")),
            stream_port=int(server_raw.get("stream_port", 8001)),
            stream_fps=int(server_raw.get("stream_fps", 12)),
            jpeg_quality=int(server_raw.get("jpeg_quality", 80)),
        ),
        logging=LoggingConfig(
            level=str(logging_raw.get("level", "INFO")),
            csv_fallback=bool(logging_raw.get("csv_fallback", True)),
            screenshot_dir=str(logging_raw.get("screenshot_dir", "data/screenshots")),
        ),
        channels=_build_channels(raw, camera_raw, desks, server_raw),
        raw=raw,
    )