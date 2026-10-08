from __future__ import annotations

import argparse
import csv
import logging
import os
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|timeout;5000000|stimeout;5000000",
)

from . import db
from .config import AppConfig, CameraChannel, load_config
from .detector import DeskAssigner, Detection, PersonDetector, draw_rois, face_region
from .enroll import EnrollmentSession
from .face_id import FaceMatch, TrackIdentityRegistry, build_identifier
from .pose import PoseEstimator, Skeleton, clip_to_bbox, smooth_skeleton, validate_skeleton
from .state_machine import DeskSpec, Event, Observation, PresenceEngine, Thresholds

LOGGER = logging.getLogger("presence.recorder")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DbSink:
    def __init__(self, camera_id: int, min_session_sec: int, desk_db_ids: dict[int, int]) -> None:
        self._camera_id = camera_id
        self._min_session_sec = min_session_sec
        self._desk_db_ids = desk_db_ids

    def open_session(
        self, desk_id: int, employee_id: int | None, track_id: int | None, sit_start: datetime
    ) -> int | None:
        return db.open_session(
            self._desk_db_ids.get(desk_id, desk_id), self._camera_id, employee_id, track_id, sit_start
        )

    def close_session(
        self, session_id: int, sit_end: datetime, away_sec: int, away_count: int
    ) -> tuple[str, int]:
        return db.close_session(session_id, sit_end, away_sec, away_count, self._min_session_sec)

    def set_session_identity(self, session_id: int, employee_id: int) -> None:
        db.update_session_identity(session_id, employee_id)


class EnrollStartRequest(BaseModel):
    name: str
    employee_no: str | None = None
    dept: str | None = None
    count: int = 10


@dataclass
class DeskView:
    desk_id: int
    label: str
    roi: tuple[float, float, float, float]
    db_id: int


class Recorder:
    def __init__(
        self,
        cfg: AppConfig,
        source: Any | None = None,
        backend: str = "sface",
        preview: bool = True,
        save_frames: bool = True,
        stream: bool = True,
        channel: CameraChannel | None = None,
    ) -> None:
        self._cfg = cfg
        self._channel = channel
        self._source = cfg.camera.source if source is None else source
        self._backend = backend
        self._preview = preview and not _is_file_source(self._source)
        self._save_frames = save_frames
        self._stream_enabled = stream
        self._tz = cfg.local_tz()
        self._stop = False
        self._capture: cv2.VideoCapture | None = None
        self._is_network = False
        self._reconnect_attempts = 0
        self._camera_id = 0
        self._engine: PresenceEngine | None = None
        self._detector: PersonDetector | None = None
        self._identifier: Any = None
        self._registry = TrackIdentityRegistry(votes_required=cfg.face.votes_required)
        self._csv_path = PROJECT_ROOT / "data" / "events.csv"
        self._frame_path = PROJECT_ROOT / "data" / "last_frame.jpg"
        self._reload_flag = PROJECT_ROOT / "data" / "reload_faces.flag"
        self._shot_dir = PROJECT_ROOT / cfg.logging.screenshot_dir
        self._last_publish = 0.0
        self._last_snapshot = 0.0
        self._fps_smooth = 0.0
        self._model_fps = 0.0
        self._frame_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_frame: Any = None
        self._stream_server: Any = None
        self._stream_thread: threading.Thread | None = None
        self._desks = DeskAssigner(cfg.presence.desk_switch_grace_sec)
        self._pose: PoseEstimator | None = None
        self._skeletons: dict[int, Skeleton] = {}
        self._skeleton_state: dict[int, Skeleton] = {}
        self._skeleton_miss: dict[int, int] = {}
        self._postures: dict[int, str] = {}
        self._pose_fps = 0.0
        self._last_pipeline_ms = 0.0
        self._loop_start = time.monotonic()
        self._loop_frames = 0
        self._enroll: EnrollmentSession | None = None
        self._enroll_last: dict[str, Any] | None = None

    def _now(self) -> datetime:
        return datetime.now(self._tz)

    def _request_stop(self, *_args: Any) -> None:
        LOGGER.info("stop signal received, closing sessions")
        self._stop = True

    def run(self) -> int:
        cfg = self._cfg
        self._shot_dir.mkdir(parents=True, exist_ok=True)
        channel = self._channel
        if channel is not None:
            cfg.camera.id = channel.id
            cfg.camera.name = channel.name
            cfg.camera.source = self._source = channel.source
            cfg.camera.width = channel.width
            cfg.camera.height = channel.height
            cfg.camera.fps = channel.fps
            cfg.camera.fourcc = channel.fourcc
            cfg.camera.backend = channel.backend
            cfg.server.stream_port = channel.stream_port
            cfg.desks = channel.desks
        self._camera_id = db.ensure_camera(cfg.camera.id, cfg.camera.name, str(self._source))
        db.sync_desks(
            self._camera_id,
            [{"label": d.label, "roi": d.roi, "employee_id": d.employee_id} for d in cfg.desks],
        )
        db_rows = db.list_desks_for_camera(self._camera_id)
        by_label = {str(row["label"]): row for row in db_rows}
        desks: list[DeskView] = []
        for desk in cfg.desks:
            row = by_label.get(desk.label)
            desks.append(
                DeskView(
                    desk_id=desk.id,
                    label=desk.label,
                    roi=tuple(row["roi"]) if row else desk.roi,
                    db_id=int(row["id"]) if row else desk.id,
                )
            )
            if row is None:
                LOGGER.error("kursi '%s' tidak ada di database, lewati", desk.label)
        self._desk_db_ids = {d.desk_id: d.db_id for d in desks}

        self._engine = PresenceEngine(
            desks=[DeskSpec(d.desk_id, d.label) for d in desks],
            sink=DbSink(self._camera_id, cfg.presence.min_session_sec, self._desk_db_ids),
            thresholds=Thresholds(
                enter_confirm_sec=cfg.presence.enter_confirm_sec,
                leave_confirm_sec=cfg.presence.leave_confirm_sec,
                away_grace_sec=cfg.presence.away_grace_sec,
                min_session_sec=cfg.presence.min_session_sec,
            ),
            camera_id=self._camera_id,
            clock=self._now,
        )
        self._restore_sessions()

        LOGGER.info("loading detector (%s on %s)", cfg.detection.model, cfg.detection.device)
        self._detector = PersonDetector(cfg.detection, PROJECT_ROOT)
        self._unified = bool(self._detector.has_keypoints)
        if self._unified:
            LOGGER.info("mode terpadu: satu model untuk deteksi + pelacakan + stickman")
        elif cfg.pose.enabled:
            try:
                self._pose = PoseEstimator(cfg.pose, PROJECT_ROOT)
                LOGGER.info("stickman aktif (model=%s on %s)", cfg.pose.model, self._pose.device)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("stickman dinonaktifkan: %s", exc)
                self._pose = None

        if cfg.face.enabled:
            try:
                self._identifier = build_identifier(cfg.face, db.load_known_faces(), backend=self._backend)
                LOGGER.info(
                    "face backend=%s, known faces=%d, threshold=%.2f",
                    self._identifier.backend,
                    self._identifier._matrix.shape[0],
                    cfg.face.similarity_threshold,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("face identification disabled: %s", exc)
                self._identifier = None

        signal.signal(signal.SIGINT, self._request_stop)
        try:
            signal.signal(signal.SIGTERM, self._request_stop)
        except (ValueError, AttributeError):
            pass

        self._start_stream_server()

        try:
            return self._loop(desks)
        finally:
            self._shutdown()

    def _restore_sessions(self) -> None:
        assert self._engine is not None
        reverse = {db_id: desk_id for desk_id, db_id in self._desk_db_ids.items()}
        for row in db.list_desks_for_camera(self._camera_id):
            db_id = int(row["id"])
            engine_id = reverse.get(db_id)
            if engine_id is None:
                continue
            session = db.get_open_session(db_id)
            if session is None:
                continue
            self._engine.restore_open_session(
                desk_id=engine_id,
                session_id=int(session["id"]),
                employee_id=session["employee_id"],
                employee_name=row["employee_name"],
                sit_start=session["sit_start"].astimezone(self._tz),
                away_sec=int(session["away_sec"] or 0),
                away_count=int(session["away_count"] or 0),
            )
            LOGGER.info("restored open session %s for desk %s", session["id"], row["label"])

    def _open_capture(self) -> bool:
        cfg = self._cfg.camera
        backends = {
            "any": cv2.CAP_ANY,
            "dshow": cv2.CAP_DSHOW,
            "msmf": cv2.CAP_MSMF,
            "ffmpeg": cv2.CAP_FFMPEG,
        }
        backend = backends.get(cfg.backend.lower(), cv2.CAP_ANY)
        is_network = isinstance(self._source, str) and self._source.lower().startswith(
            ("rtsp://", "rtmp://", "http://", "https://")
        )
        if is_network:
            LOGGER.info("membuka stream jaringan: %s", _mask_source(self._source))
            capture = cv2.VideoCapture(self._source, cv2.CAP_FFMPEG)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
        else:
            LOGGER.info("opening source %s (backend=%s)", self._source, cfg.backend)
            capture = cv2.VideoCapture(self._source, backend)
            if not capture.isOpened() and backend != cv2.CAP_ANY:
                capture.release()
                capture = cv2.VideoCapture(self._source)
            if cfg.fourcc and len(cfg.fourcc) == 4:
                capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
            capture.set(cv2.CAP_PROP_FPS, cfg.fps)
        if not capture.isOpened():
            LOGGER.error("cannot open source %s", _mask_source(self._source))
            return False

        actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
        LOGGER.info(
            "camera aktif: %sx%s @ %.1f fps%s",
            actual_w,
            actual_h,
            actual_fps,
            f" (diminta {cfg.width}x{cfg.height})" if not is_network else "",
        )
        self._capture = capture
        self._is_network = is_network
        return True

    def _sleep_interruptible(self, seconds: float) -> bool:
        """Tidur dalam potongan kecil agar Ctrl+C tetap responsif."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._stop:
                return False
            time.sleep(0.2)
        return not self._stop

    def _reconnect(self) -> bool:
        """Sambung ulang stream jaringan, berulang sampai berhasil.

        Kabel LAN ke switch PoE sering goyah. Dulu Recorder menyerah setelah satu
        percobaan, sehingga kamera yang kembali online 5 detik kemudian tetap
        tidak tertangkap sampai proses dimulai ulang.
        """
        while not self._stop:
            if self._capture is not None:
                try:
                    self._capture.release()
                except Exception:  # noqa: BLE001
                    pass
            self._capture = None
            delay = min(30.0, 2.0 * (2 ** min(self._reconnect_attempts, 4)))
            self._reconnect_attempts += 1
            LOGGER.warning(
                "stream terputus, mencoba sambung lagi dalam %.0f dtk (percobaan %d)",
                delay,
                self._reconnect_attempts,
            )
            if not self._sleep_interruptible(delay):
                return False
            if self._open_capture():
                self._reconnect_attempts = 0
                LOGGER.info("stream tersambung kembali")
                return True
        return False

    def _loop(self, desks: list[DeskView]) -> int:
        assert self._engine is not None and self._detector is not None
        cfg = self._cfg
        attempt = 0
        while not self._stop:
            if self._open_capture():
                break
            if not _is_network_source(self._source):
                return 2
            attempt += 1
            delay = min(30.0, 2.0 * (2 ** min(attempt, 4)))
            LOGGER.warning(
                "stream belum tersedia, mencoba lagi dalam %.0f dtk (percobaan %d)",
                delay,
                attempt,
            )
            self._stop = False
            time.sleep(delay)
        if self._stop:
            return 0

        desk_tuples = [(d.desk_id, d.label, d.roi) for d in desks]
        read_failures = 0
        # Untuk stream jaringan, tiap read yang gagal bisa blocking sampai
        # READ_TIMEOUT_MSEC (10 dtk), jadi ambang 30 berarti ~5 menit sebelum
        # reconnect. 5 kegagalan cukup untuk memutuskan stream sudah putus.
        max_failures = 5 if _is_network_source(self._source) else 30
        frame: Any = None
        LOGGER.info("recording started (q=quit)")

        while not self._stop:
            ok, frame = self._capture.read()
            if not ok or frame is None:
                read_failures += 1
                if _is_file_source(self._source):
                    LOGGER.info("end of file reached")
                    break
                if read_failures > max_failures:
                    if self._is_network and not self._stop and self._reconnect():
                        continue
                    LOGGER.error("too many failed reads, stopping")
                    break
                time.sleep(0.05)
                continue
            read_failures = 0
            frame_start = time.monotonic()
            self._loop_frames += 1
            elapsed = time.monotonic() - self._loop_start
            if elapsed >= 1.0:
                self._fps_smooth = self._loop_frames / elapsed
                self._loop_frames = 0
                self._loop_start = time.monotonic()

            result = self._detector.track(frame)
            if result.fps > 0:
                self._model_fps = result.fps if self._model_fps == 0 else 0.8 * self._model_fps + 0.2 * result.fps

            self._update_pose(frame, result, cfg.pose.every_n_frames)
            anchors: dict[int, tuple[float, float]] = {}
            for track_id, skeleton in self._skeletons.items():
                anchors[track_id] = skeleton.torso_center()

            per_desk = self._desks.assign(
                result.detections,
                desk_tuples,
                frame.shape,
                cfg.presence.desk_overlap_ratio,
                time.monotonic(),
                anchors=anchors,
            )
            observations: dict[int, Observation] = {}
            for desk_id, detection in per_desk.items():
                employee_id, employee_name, similarity = self._identify(detection, result.frame_id, frame)
                observations[desk_id] = Observation(
                    desk_id=desk_id,
                    employee_id=employee_id,
                    employee_name=employee_name,
                    track_id=detection.track_id,
                    similarity=similarity,
                )
            self._registry.prune({d.track_id for d in result.detections})

            self._maybe_enroll(frame, result.detections)

            events = self._engine.update(observations)
            if events:
                self._handle_events(events, frame)

            self._publish_frame(frame, observations, desk_tuples)
            self._last_pipeline_ms = (time.monotonic() - frame_start) * 1000.0

            now_ts = time.monotonic()
            if now_ts - self._last_publish >= cfg.server.poll_seconds:
                self._last_publish = now_ts
                self._publish_status()

            if self._preview:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")) or key == 27:
                    self._stop = True
        return 0

    def _enroll_start(self, name: str, employee_no: str | None, dept: str | None, count: int) -> dict:
        with self._frame_lock:
            self._enroll = EnrollmentSession(
                name=name,
                employee_no=employee_no or None,
                dept=dept or None,
                target_count=max(3, min(30, int(count))),
            )
        self._enroll_last = None
        LOGGER.info("enrollment dimulai: %s (target %d sampel)", name, self._enroll.target_count)
        return self._enroll.progress()

    def _enroll_status(self) -> dict:
        if self._enroll is not None:
            payload = self._enroll.progress()
            payload["last_result"] = self._enroll_last
            return payload
        return {"status": "idle", "last_result": self._enroll_last}

    def _enroll_cancel(self) -> dict:
        if self._enroll is not None:
            self._enroll.cancel()
            LOGGER.info("enrollment dibatalkan")
            self._enroll = None
        return self._enroll_status()

    def _enroll_commit(self) -> dict:
        session = self._enroll
        if session is None:
            raise ValueError("tidak ada sesi enrollment aktif")
        if not session.ready():
            raise ValueError(f"sampel belum cukup ({session.count}/{session.target_count})")
        embedding = session.finish()
        try:
            employee_id = db.add_employee(session.name, session.employee_no, session.dept, None)
        except ValueError as exc:
            session.status = "running"
            raise ValueError(str(exc)) from exc
        db.save_employee_embedding(employee_id, embedding, session.count)
        flag = self._reload_flag
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.touch()
        similarities = [s.similarity for s in session.samples if s.similarity is not None]
        result = {
            "employee_id": employee_id,
            "name": session.name,
            "samples": session.count,
            "dim": int(embedding.shape[0]),
            "avg_similarity": round(float(np.mean(similarities)), 3) if similarities else None,
        }
        self._enroll_last = result
        self._enroll = None
        LOGGER.info("enrollment selesai: id=%s samples=%s", employee_id, result["samples"])
        return result

    def _maybe_enroll(self, frame: Any, detections: list[Detection]) -> None:
        session = self._enroll
        if session is None or session.status != "running" or self._identifier is None:
            return
        if session.count >= session.target_count:
            return
        if not detections:
            session.last_error = "tidak ada orang terdeteksi"
            return
        cfg = self._cfg.face
        detection = max(detections, key=lambda d: d.conf)
        crop = face_region(
            frame,
            detection.bbox,
            cfg.region_ratio,
            cfg.region_top_offset,
            cfg.widen,
            cfg.min_face_px,
            cfg.crop_max_width,
        )
        if crop is None:
            session.last_error = "area wajah terlalu kecil"
            return
        faces = self._identifier.detect(crop, cfg.min_face_px)
        if not faces:
            session.last_error = "wajah belum terdeteksi"
            return
        biggest = max(faces, key=lambda c: c.shape[0] * c.shape[1])
        aligned = cv2.resize(biggest, (112, 112), interpolation=cv2.INTER_LINEAR)
        vector = self._identifier.embed(aligned)
        similarity = None
        if self._identifier._matrix.shape[0]:
            scores = self._identifier._matrix @ vector
            similarity = float(np.max(scores))
        if session.add_sample(vector, similarity):
            session.last_error = None

    def _update_pose(self, frame: Any, result: Any, every_n_frames: int) -> None:
        cfg = self._cfg.pose
        skeletons_out: dict[int, Skeleton] = {}
        postures: dict[int, str] = {}
        seen_tracks: set[int] = set()

        if self._unified:
            due = every_n_frames <= 0 or result.frame_id % every_n_frames == 0
            candidates = [d for d in result.detections if d.keypoints is not None]
        else:
            due = every_n_frames <= 0 or result.frame_id % every_n_frames == 0
            candidates = []
            if due and self._pose is not None:
                candidates = self._pose.estimate(frame)

        for detection in result.detections:
            track_id = detection.track_id
            if track_id < 0:
                continue
            stored = self._skeleton_state.get(track_id)
            missed = self._skeleton_miss.get(track_id, 0)
            accepted = False
            if due and candidates:
                if self._unified:
                    nearest = candidates[0] if len(candidates) == 1 else min(
                        candidates,
                        key=lambda d: float(
                            np.linalg.norm(
                                np.array((d.center[0], d.center[1]))
                                - np.array((detection.center[0], detection.center[1]))
                            )
                        ),
                    )
                    source = nearest.keypoints
                else:
                    cx, cy = detection.center
                    nearest = min(
                        candidates,
                        key=lambda s: float(np.linalg.norm(np.array(s.center()) - np.array([cx, cy]))),
                    )
                    source = nearest.keypoints
                skeleton = Skeleton(keypoints=np.asarray(source, dtype=np.float32), score=detection.conf)
                clipped, in_box_ratio = clip_to_bbox(skeleton, detection.bbox, cfg.bbox_margin)
                if cfg.validate_limbs:
                    clipped = validate_skeleton(
                        clipped,
                        min_conf=cfg.keypoint_confidence,
                        limb_ratio_range=(cfg.limb_ratio_min, cfg.limb_ratio_max),
                    )
                if in_box_ratio >= cfg.min_inbox_ratio and clipped.valid_count(
                    cfg.keypoint_confidence
                ) >= cfg.draw_min_keypoints:
                    clipped = smooth_skeleton(clipped, stored, cfg.smoothing)
                    self._skeleton_state[track_id] = clipped
                    self._skeleton_miss[track_id] = 0
                    skeletons_out[track_id] = clipped
                    accepted = True
            if not accepted and stored is not None and missed < cfg.hold_frames:
                skeletons_out[track_id] = stored
                accepted = True
            if not accepted:
                self._skeleton_miss[track_id] = missed + 1
                continue
            seen_tracks.add(track_id)
            reference = skeletons_out[track_id]
            postures[track_id] = reference.posture(
                min_conf=cfg.keypoint_confidence,
                sitting_knee_ratio=cfg.sitting_knee_ratio,
                standing_knee_ratio=cfg.standing_knee_ratio,
                min_keypoints=cfg.min_keypoints,
            )
        for stale in [t for t in self._skeleton_state if t not in seen_tracks]:
            self._skeleton_state.pop(stale, None)
            self._skeleton_miss.pop(stale, None)
        self._skeletons = skeletons_out
        self._postures = postures

    def _identify(
        self, detection: Detection, frame_id: int, frame: Any
    ) -> tuple[int | None, str | None, float | None]:
        cfg = self._cfg
        if self._identifier is None or detection.track_id < 0:
            return None, None, None
        is_new = detection.track_id not in self._registry.seen_tracks
        due = cfg.face.run_every_n_frames <= 0 or frame_id % cfg.face.run_every_n_frames == 0
        if not (due or is_new):
            employee_id, employee_name = self._registry.identity(detection.track_id)
            return employee_id, employee_name, None
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
            employee_id, employee_name = self._registry.identity(detection.track_id)
            return employee_id, employee_name, None
        match: FaceMatch = self._identifier.match(crop)
        employee_id, employee_name, similarity = self._registry.observe(detection.track_id, match)
        return employee_id, employee_name, similarity

    def _handle_events(self, events: list[Event], frame: Any) -> None:
        rows: list[dict[str, Any]] = []
        for event in events:
            LOGGER.info(
                "%s desk=%s employee=%s session=%s duration=%ss away=%ss",
                event.type,
                event.desk_label,
                event.employee_name or "-",
                event.session_id,
                event.duration_sec,
                event.away_sec,
            )
            try:
                db.record_event(
                    event.type,
                    desk_id=self._desk_db_ids.get(event.desk_id, event.desk_id),
                    employee_id=event.employee_id,
                    session_id=event.session_id,
                    note=event.note,
                    ts=event.ts,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("failed to store event %s: %s", event.type, exc)
            rows.append(
                {
                    "ts": event.ts.isoformat(),
                    "type": event.type,
                    "desk_id": event.desk_id,
                    "desk_label": event.desk_label,
                    "employee_id": event.employee_id,
                    "employee_name": event.employee_name or "",
                    "session_id": event.session_id,
                    "duration_sec": event.duration_sec,
                    "away_sec": event.away_sec,
                    "note": event.note or "",
                }
            )
            if self._save_frames and frame is not None:
                self._snapshot(event, frame)
        self._append_csv(rows)

    def _append_csv(self, rows: list[dict[str, Any]]) -> None:
        if not rows or not self._cfg.logging.csv_fallback:
            return
        exists = self._csv_path.exists()
        try:
            with open(self._csv_path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                if not exists:
                    writer.writeheader()
                writer.writerows(rows)
        except OSError as exc:
            LOGGER.error("csv fallback failed: %s", exc)

    def _snapshot(self, event: Event, frame: Any) -> None:
        stamp = event.ts.strftime("%Y%m%d-%H%M%S")
        path = self._shot_dir / f"{stamp}-{event.type.lower()}-desk{event.desk_id}.jpg"
        try:
            cv2.imwrite(str(path), frame.copy())
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("screenshot failed: %s", exc)

    def _start_stream_server(self) -> None:
        cfg = self._cfg.server
        if not self._stream_enabled:
            return
        from fastapi import FastAPI, HTTPException, Response
        from fastapi.responses import StreamingResponse
        from uvicorn import Config, Server

        # Pre-flight: port sudah dipakai instance lain? uvicorn bind di dalam
        # thread, jadi error-nya hanya muncul di log dan recorder tetap jalan
        # tanpa MJPEG (deteksi ganda ke kamera yang sama). Stop di sini saja.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((cfg.stream_host or "0.0.0.0", cfg.stream_port))
            except OSError as exc:
                raise RuntimeError(
                    f"port MJPEG {cfg.stream_port} sudah dipakai proses lain "
                    f"({exc}). Kemungkinan ada instance recorder ganda yang masih "
                    f"jalan - hentikan dulu, jangan biarkan 2 recorder menarik "
                    f"kamera yang sama (batas klien kamera akan habis)."
                ) from exc

        recorder = self

        app = FastAPI(title="Employee Presence Stream", docs_url=None, redoc_url=None)

        @app.get("/stream")
        def stream():
            return StreamingResponse(
                recorder._mjpeg(),
                media_type="multipart/x-mixed-replace; boundary=frame",
headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
            )

        @app.get("/snapshot.jpg")
        def snapshot():
            with recorder._frame_lock:
                frame = recorder._latest_frame
            if frame is None:
                return Response(content=b"", media_type="image/jpeg")
            ok, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, recorder._cfg.server.jpeg_quality]
            )
            return Response(content=buffer.tobytes() if ok else b"", media_type="image/jpeg")

        @app.get("/health")
        def health():
            return {"status": "ok", "streaming": recorder._latest_frame is not None}

        @app.get("/enroll/status")
        def enroll_status():
            return recorder._enroll_status()

        @app.post("/enroll/start")
        def enroll_start(payload: EnrollStartRequest):
            if not payload.name.strip():
                raise HTTPException(status_code=400, detail="nama wajib diisi")
            if recorder._enroll is not None:
                raise HTTPException(status_code=409, detail="sudah ada enrollment aktif")
            return recorder._enroll_start(
                payload.name.strip(), payload.employee_no, payload.dept, payload.count
            )

        @app.post("/enroll/commit")
        def enroll_commit():
            try:
                return recorder._enroll_commit()
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

        @app.post("/enroll/cancel")
        def enroll_cancel():
            return recorder._enroll_cancel()

        server = Server(Config(app, host=cfg.stream_host, port=cfg.stream_port, log_level="warning"))
        self._stream_server = server
        self._stream_thread = threading.Thread(target=server.run, daemon=True, name="mjpeg-server")
        self._stream_thread.start()
        LOGGER.info("MJPEG stream: http://%s:%d/stream", cfg.stream_host, cfg.stream_port)

    def _mjpeg(self):
        interval = 1.0 / max(1, self._cfg.server.stream_fps)
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n"
        last_sent = 0.0
        while not self._stop:
            with self._frame_lock:
                frame = self._latest_frame
            now = time.monotonic()
            if frame is None or now - last_sent < interval:
                time.sleep(0.005)
                continue
            last_sent = now
            ok, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._cfg.server.jpeg_quality]
            )
            if not ok:
                continue
            data = buffer.tobytes()
            yield boundary + f"Content-Length: {len(data)}\r\n\r\n".encode() + data + b"\r\n"
        time.sleep(0.2)

    def _stop_stream_server(self) -> None:
        if self._stream_server is not None:
            self._stream_server.should_exit = True
            if self._stream_thread is not None:
                self._stream_thread.join(timeout=5)
            self._stream_server = None
            self._stream_thread = None

    def _render(
        self,
        frame: Any,
        desk_tuples: list[tuple[int, str, tuple[float, float, float, float]]],
        observations: dict[int, Observation],
    ) -> Any:
        assert self._engine is not None
        overlay = draw_rois(frame.copy(), desk_tuples)
        if self._pose is not None and self._skeletons:
            self._pose.draw(overlay, list(self._skeletons.values()))
        for index, state in enumerate(self._engine.states.values()):
            obs = observations.get(state.desk_id)
            now = self._now()
            name = state.employee_name or (obs.employee_name if obs else None) or "Tidak dikenal"
            if state.status.value == "PRESENT":
                seconds = state.duration_sec(now)
                text = f"{state.label}: {name} | duduk {seconds // 60}m {seconds % 60}s"
                color = (60, 220, 60)
            elif state.session_id is None and state.pending_since is not None and (now - state.pending_since).total_seconds() > 0.3:
                waited = int((now - state.pending_since).total_seconds())
                need = self._cfg.presence.enter_confirm_sec
                text = f"{state.label}: {name} | terdeteksi, konfirmasi {waited}/{need}s"
                color = (60, 200, 255)
            else:
                text = f"{state.label}: kosong"
                color = (90, 90, 230)
            cv2.putText(overlay, text, (12, 46 + 26 * index),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(
            overlay,
            f"loop {self._fps_smooth:4.0f} fps | model {self._model_fps:4.0f} fps | {self._last_pipeline_ms:4.0f} ms/frame",
            (12, overlay.shape[0] - 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (215, 215, 215),
            2,
        )
        if self._cfg.pose.posture_labels:
            labels: list[str] = []
            for track_id, skeleton in self._skeletons.items():
                posture = self._postures.get(track_id, "-")
                cx, cy = skeleton.center()
                cv2.putText(
                    overlay,
                    f"#{track_id} {posture}",
                    (int(cx) - 30, int(cy) - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (60, 220, 120) if posture == "DUDUK" else (255, 170, 60),
                    2,
                )
                labels.append(f"#{track_id}:{posture}")
            if labels:
                cv2.putText(
                    overlay,
                    "stickman " + "  ".join(labels),
                    (12, overlay.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    2,
                )
        return overlay

    def _publish_frame(self, frame: Any, observations: dict[int, Observation], desk_tuples) -> None:
        overlay = self._render(frame, desk_tuples, observations)
        with self._frame_lock:
            self._latest_frame = overlay
            self._latest_jpeg = None
        if self._preview:
            cv2.imshow("employee-presence", overlay)
        now = time.monotonic()
        if self._save_frames and now - self._last_snapshot >= 2.0:
            self._last_snapshot = now
            try:
                cv2.imwrite(str(self._frame_path), overlay)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("frame save failed: %s", exc)

    def _publish_status(self) -> None:
        assert self._engine is not None
        self._maybe_reload_faces()
        snapshot = self._engine.snapshot(self._now())
        rows = []
        for row in snapshot:
            rows.append(
                {
                    "desk_id": self._desk_db_ids.get(row["desk_id"], row["desk_id"]),
                    "label": row["label"],
                    "status": row["status"],
                    "employee_id": row["employee_id"],
                    "employee_name": row["employee_name"],
                    "track_id": row["track_id"],
                    "sit_since": row["sit_since"],
                    "duration_sec": row["duration_sec"],
                    "away_sec": row["away_sec"],
                    "session_id": row["session_id"],
                    "fps": round(self._fps_smooth, 2),
                }
            )
        if not rows:
            return
        try:
            db.upsert_live_status_many(rows)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("live status update failed: %s", exc)

    def _maybe_reload_faces(self) -> None:
        flag = self._reload_flag
        if self._identifier is None or not flag.exists():
            return
        try:
            flag.unlink()
            self._identifier.set_known(db.load_known_faces())
            self._registry = TrackIdentityRegistry(votes_required=self._cfg.face.votes_required)
            LOGGER.info("known faces reloaded: %d", self._identifier._matrix.shape[0])
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("face reload failed: %s", exc)

    def _shutdown(self) -> None:
        assert self._engine is not None
        self._stop = True
        self._stop_stream_server()
        events = self._engine.flush(self._now())
        self._handle_events(events, None)
        try:
            db.close_all_open_sessions(self._now())
            db.clear_live_status()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("cleanup failed: %s", exc)
        if self._capture is not None:
            self._capture.release()
        cv2.destroyAllWindows()
        LOGGER.info("recording stopped")


NETWORK_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://")


def _is_network_source(source: Any) -> bool:
    return isinstance(source, str) and source.lower().startswith(NETWORK_SCHEMES)


def _mask_source(source: Any) -> str:
    text = str(source)
    if "@" in text and "//" in text:
        scheme, rest = text.split("//", 1)
        credentials, host = rest.split("@", 1)
        return f"{scheme}//{credentials.split(':')[0]}:***@{host}"
    return text


def _is_file_source(source: Any) -> bool:
    """True HANYA untuk file video di disk.

    Dulu ini `not source.isdigit()`, sehingga setiap URL RTSP ikut dianggap
    file. Akibatnya begitu satu read gagal, recorder mencetak "end of file
    reached" lalu berhenti permanen - tidak pernah mencoba reconnect, padahal
    kameranya masih hidup. Thread MJPEG tetap melayani frame basi sehingga
    dashboard tampak normal padahal deteksi sudah mati.
    """
    if not isinstance(source, str):
        return False
    if _is_network_source(source):
        return False
    text = source.strip()
    if not text:
        return False
    if text.isdigit():
        return False  # index kamera (0, 1, ...)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rekam kehadiran karyawan di kursi kerja")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--source", default=None, help="0|1 untuk kamera, path file untuk video")
    parser.add_argument("--backend", default="sface", choices=["sface", "insightface"])
    parser.add_argument("--no-preview", action="store_true", help="jangan tampilkan window OpenCV")
    parser.add_argument("--no-save-frames", action="store_true", help="jangan simpan frame/screenshot")
    parser.add_argument("--no-stream", action="store_true", help="jangan jalankan server MJPEG")
    parser.add_argument("--camera-id", default=None, help="pilih kamera dari daftar cameras:")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    source: Any = args.source if args.source is None else int(args.source) if args.source.isdigit() else args.source
    channel = None
    if args.camera_id:
        channel = next((c for c in cfg.channels if c.id == args.camera_id), None)
        if channel is None:
            available = ", ".join(c.id for c in cfg.channels)
            print(f"kamera '{args.camera_id}' tidak ada di config. Tersedia: {available}")
            return 2
        source = None
    recorder = Recorder(
        cfg,
        source=source,
        channel=channel,
        backend=args.backend,
        preview=not args.no_preview,
        save_frames=not args.no_save_frames,
        stream=not args.no_stream,
    )
    return recorder.run()


if __name__ == "__main__":
    sys.exit(main())