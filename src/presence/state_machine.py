from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Protocol


class Status(str, Enum):
    PRESENT = "PRESENT"
    AWAY = "AWAY"


@dataclass(frozen=True)
class Observation:
    desk_id: int
    employee_id: int | None = None
    employee_name: str | None = None
    track_id: int | None = None
    similarity: float | None = None


@dataclass
class Event:
    type: str
    desk_id: int
    desk_label: str
    ts: datetime
    employee_id: int | None = None
    employee_name: str | None = None
    session_id: int | None = None
    duration_sec: int = 0
    away_sec: int = 0
    away_count: int = 0
    note: str | None = None


@dataclass
class DeskState:
    desk_id: int
    label: str
    status: Status = Status.AWAY
    employee_id: int | None = None
    employee_name: str | None = None
    track_id: int | None = None
    session_id: int | None = None
    sit_since: datetime | None = None
    last_seen_at: datetime | None = None
    pending_since: datetime | None = None
    pending_gap_start: datetime | None = None
    away_since: datetime | None = None
    away_total_sec: int = 0
    away_count: int = 0
    current_away_sec: int = 0

    def duration_sec(self, now: datetime) -> int:
        if self.sit_since is None:
            return 0
        return max(0, int((now - self.sit_since).total_seconds()))

    def live_away_sec(self, now: datetime) -> int:
        if self.away_since is None:
            return self.away_total_sec
        return self.away_total_sec + max(0, int((now - self.away_since).total_seconds()))


class SessionSink(Protocol):
    def open_session(self, desk_id: int, employee_id: int | None, track_id: int | None, sit_start: datetime) -> int: ...

    def close_session(self, session_id: int, sit_end: datetime, away_sec: int, away_count: int) -> tuple[str, int]: ...

    def set_session_identity(self, session_id: int, employee_id: int) -> None: ...


@dataclass
class DeskSpec:
    desk_id: int
    label: str


@dataclass
class Thresholds:
    enter_confirm_sec: int = 10
    leave_confirm_sec: int = 10
    away_grace_sec: int = 60
    min_session_sec: int = 300


class PresenceEngine:
    def __init__(
        self,
        desks: list[DeskSpec],
        sink: SessionSink,
        thresholds: Thresholds,
        camera_id: int,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._sink = sink
        self._thresholds = thresholds
        self._camera_id = camera_id
        self._clock = clock or (lambda: datetime.now().astimezone())
        self.states: dict[int, DeskState] = {
            d.desk_id: DeskState(desk_id=d.desk_id, label=d.label) for d in desks
        }

    def restore_open_session(
        self,
        desk_id: int,
        session_id: int,
        employee_id: int | None,
        employee_name: str | None,
        sit_start: datetime,
        away_sec: int = 0,
        away_count: int = 0,
    ) -> None:
        state = self.states.get(desk_id)
        if state is None or state.session_id is not None:
            return
        state.session_id = session_id
        state.employee_id = employee_id
        state.employee_name = employee_name
        state.sit_since = sit_start
        state.last_seen_at = self._clock()
        state.status = Status.PRESENT
        state.away_total_sec = away_sec
        state.away_count = away_count

    def update(self, observations: dict[int, Observation]) -> list[Event]:
        now = self._clock()
        events: list[Event] = []
        for desk_id, state in self.states.items():
            obs = observations.get(desk_id)
            events.extend(self._update_desk(state, obs, now))
        return events

    def _update_desk(self, state: DeskState, obs: Observation | None, now: datetime) -> list[Event]:
        events: list[Event] = []
        th = self._thresholds

        if obs is not None:
            if state.pending_since is None:
                state.pending_since = now
            state.pending_gap_start = None
            state.last_seen_at = now
            if obs.track_id is not None:
                state.track_id = obs.track_id
            if obs.employee_id is not None and obs.employee_id != state.employee_id:
                state.employee_id = obs.employee_id
                state.employee_name = obs.employee_name
                if state.session_id is not None:
                    self._sink.set_session_identity(state.session_id, obs.employee_id)
                events.append(
                    Event(
                        type="IDENTIFIED",
                        desk_id=state.desk_id,
                        desk_label=state.label,
                        ts=now,
                        employee_id=obs.employee_id,
                        employee_name=obs.employee_name,
                        session_id=state.session_id,
                        note=f"similarity={obs.similarity:.3f}" if obs.similarity else None,
                    )
                )
        elif state.pending_since is not None:
            if state.pending_gap_start is None:
                state.pending_gap_start = now
            elif (now - state.pending_gap_start).total_seconds() > th.leave_confirm_sec:
                state.pending_since = None
                state.pending_gap_start = None

        has_session = state.session_id is not None
        gap_sec = (
            0.0
            if obs is not None or state.pending_gap_start is None
            else (now - state.pending_gap_start).total_seconds()
        )
        confirmed_present = obs is not None or (
            state.pending_since is not None and gap_sec <= th.leave_confirm_sec
        )

        if confirmed_present and not has_session:
            pending = state.pending_since
            if pending is not None and (now - pending) >= timedelta(seconds=th.enter_confirm_sec):
                session_id = self._sink.open_session(
                    state.desk_id, state.employee_id, state.track_id, pending
                )
                if session_id is not None:
                    state.session_id = session_id
                    state.sit_since = pending
                    state.status = Status.PRESENT
                    state.away_since = None
                    state.away_total_sec = 0
                    state.away_count = 0
                    state.current_away_sec = 0
                    events.append(
                        Event(
                            type="SESSION_OPENED",
                            desk_id=state.desk_id,
                            desk_label=state.label,
                            ts=pending,
                            employee_id=state.employee_id,
                            employee_name=state.employee_name,
                            session_id=session_id,
                        )
                    )
            return events

        if not has_session:
            state.status = Status.AWAY
            state.current_away_sec = 0
            return events

        assert state.last_seen_at is not None
        absent_sec = max(0.0, (now - state.last_seen_at).total_seconds())

        if confirmed_present and state.status == Status.AWAY:
            if state.away_since is not None:
                state.away_total_sec += int((now - state.away_since).total_seconds())
                state.away_count += 1
                state.away_since = None
            state.current_away_sec = state.away_total_sec
            state.status = Status.PRESENT
            events.append(
                Event(
                    type="AWAY_END",
                    desk_id=state.desk_id,
                    desk_label=state.label,
                    ts=now,
                    employee_id=state.employee_id,
                    employee_name=state.employee_name,
                    session_id=state.session_id,
                    away_sec=state.away_total_sec,
                    away_count=state.away_count,
                )
            )
            return events

        if absent_sec <= th.leave_confirm_sec:
            return events

        if state.status == Status.PRESENT:
            state.status = Status.AWAY
            state.away_since = state.last_seen_at
            events.append(
                Event(
                    type="AWAY_START",
                    desk_id=state.desk_id,
                    desk_label=state.label,
                    ts=state.away_since,
                    employee_id=state.employee_id,
                    employee_name=state.employee_name,
                    session_id=state.session_id,
                )
            )
            return events

        if state.away_since is not None:
            state.away_total_sec += int((now - state.away_since).total_seconds())
            state.current_away_sec = state.away_total_sec
            state.away_count += 1
            state.away_since = None

        if absent_sec >= th.away_grace_sec:
            sit_end = state.last_seen_at + timedelta(seconds=th.away_grace_sec)
            status, duration = self._sink.close_session(
                state.session_id, sit_end, state.away_total_sec, state.away_count
            )
            events.append(
                Event(
                    type="SESSION_CLOSED" if status == "closed" else "SESSION_DISCARDED",
                    desk_id=state.desk_id,
                    desk_label=state.label,
                    ts=sit_end,
                    employee_id=state.employee_id,
                    employee_name=state.employee_name,
                    session_id=state.session_id,
                    duration_sec=duration,
                    away_sec=state.away_total_sec,
                    away_count=state.away_count,
                    note=None if status == "closed" else f"duration<{th.min_session_sec}s",
                )
            )
            state.session_id = None
            state.sit_since = None
            state.away_total_sec = 0
            state.away_count = 0
            state.current_away_sec = 0
            state.status = Status.AWAY

        return events

    def flush(self, now: datetime | None = None) -> list[Event]:
        now = now or self._clock()
        events: list[Event] = []
        for state in self.states.values():
            if state.session_id is None:
                continue
            sit_end = state.last_seen_at or now
            status, duration = self._sink.close_session(
                state.session_id, sit_end, state.away_total_sec, state.away_count
            )
            events.append(
                Event(
                    type="SESSION_CLOSED" if status == "closed" else "SESSION_DISCARDED",
                    desk_id=state.desk_id,
                    desk_label=state.label,
                    ts=sit_end,
                    employee_id=state.employee_id,
                    employee_name=state.employee_name,
                    session_id=state.session_id,
                    duration_sec=duration,
                    away_sec=state.away_total_sec,
                    away_count=state.away_count,
                    note="recorder stopped",
                )
            )
            state.session_id = None
            state.sit_since = None
            state.status = Status.AWAY
        return events

    def snapshot(self, now: datetime | None = None) -> list[dict]:
        now = now or self._clock()
        rows: list[dict] = []
        for state in self.states.values():
            rows.append(
                {
                    "desk_id": state.desk_id,
                    "label": state.label,
                    "status": state.status.value,
                    "employee_id": state.employee_id,
                    "employee_name": state.employee_name,
                    "track_id": state.track_id,
                    "session_id": state.session_id,
                    "sit_since": state.sit_since,
                    "pending_since": state.pending_since,
                    "duration_sec": state.duration_sec(now) if state.status == Status.PRESENT else 0,
                    "away_sec": state.live_away_sec(now) if state.status == Status.AWAY else state.away_total_sec,
                    "away_count": state.away_count,
                }
            )
        return rows