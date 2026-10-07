from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from presence.state_machine import (
    DeskSpec,
    Observation,
    PresenceEngine,
    Status,
    Thresholds,
)

TZ = timezone(timedelta(hours=7))


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> datetime:
        self.now = self.now + timedelta(seconds=seconds)
        return self.now


class FakeSink:
    def __init__(self, min_session_sec: int = 300) -> None:
        self.min_session_sec = min_session_sec
        self.opened: list[tuple[int, int | None, int | None, datetime]] = []
        self.closed: list[tuple[int, datetime, int, int]] = []
        self.identities: list[tuple[int, int]] = []
        self.opened_time: dict[int, datetime] = {}
        self._next_id = 100

    def open_session(self, desk_id, employee_id, track_id, sit_start):
        self._next_id += 1
        session_id = self._next_id
        self.opened.append((desk_id, employee_id, track_id, sit_start))
        self.opened_time[session_id] = sit_start
        return session_id

    def close_session(self, session_id, sit_end, away_sec, away_count):
        duration = int((sit_end - self.opened_time[session_id]).total_seconds())
        self.closed.append((session_id, sit_end, away_sec, away_count))
        status = "closed" if duration >= self.min_session_sec else "discarded"
        return status, duration

    def set_session_identity(self, session_id, employee_id):
        self.identities.append((session_id, employee_id))


TrackedSink = FakeSink


def build_engine(clock: FakeClock, sink: FakeSink, **overrides):
    thresholds = Thresholds(
        enter_confirm_sec=overrides.get("enter_confirm_sec", 10),
        leave_confirm_sec=overrides.get("leave_confirm_sec", 10),
        away_grace_sec=overrides.get("away_grace_sec", 60),
        min_session_sec=overrides.get("min_session_sec", 300),
    )
    return PresenceEngine(
        desks=[DeskSpec(1, "Kursi 1"), DeskSpec(2, "Kursi 2")],
        sink=sink,
        thresholds=thresholds,
        camera_id=1,
        clock=clock,
    )


def test_no_detection_keeps_desk_away():
    clock = FakeClock(datetime(2026, 1, 5, 9, 0, tzinfo=TZ))
    sink = TrackedSink()
    engine = build_engine(clock, sink)
    events = engine.update({})
    assert events == []
    assert engine.states[1].status is Status.AWAY
    assert sink.opened == []


def test_session_opens_after_enter_confirm():
    clock = FakeClock(datetime(2026, 1, 5, 9, 0, tzinfo=TZ))
    sink = TrackedSink()
    engine = build_engine(clock, sink)
    obs = {1: Observation(desk_id=1, employee_id=7, employee_name="Budi", track_id=3)}

    engine.update(obs)
    clock.advance(9)
    events = engine.update(obs)
    assert events == []

    clock.advance(2)
    events = engine.update(obs)
    assert [e.type for e in events] == ["SESSION_OPENED"]
    assert events[0].desk_id == 1
    assert sink.opened[0][1] == 7
    assert engine.states[1].status is Status.PRESENT


def test_gaps_within_grace_accumulate_toward_confirmation():
    clock = FakeClock(datetime(2026, 1, 5, 9, 0, tzinfo=TZ))
    sink = TrackedSink()
    engine = build_engine(clock, sink)
    obs = {1: Observation(desk_id=1, employee_id=7, employee_name="Budi", track_id=3)}

    engine.update(obs)
    clock.advance(6)
    assert engine.update({}) == []

    clock.advance(6)
    events = engine.update(obs)
    assert [e.type for e in events] == ["SESSION_OPENED"]
    assert engine.states[1].sit_since == datetime(2026, 1, 5, 9, 0, tzinfo=TZ)
    assert len(sink.opened) == 1


def test_short_absence_does_not_reset_timer():
    clock = FakeClock(datetime(2026, 1, 5, 9, 0, tzinfo=TZ))
    sink = TrackedSink(min_session_sec=60)
    engine = build_engine(clock, sink, enter_confirm_sec=5, leave_confirm_sec=10, away_grace_sec=60)
    obs = {1: Observation(desk_id=1, employee_id=7, employee_name="Budi", track_id=3)}

    for _ in range(2):
        engine.update(obs)
        clock.advance(5)
    engine.update(obs)
    assert len(sink.opened) == 1
    session_id = engine.states[1].session_id
    sit_since = engine.states[1].sit_since

    clock.advance(8)
    engine.update({})
    assert engine.states[1].status is Status.PRESENT
    clock.advance(2)
    engine.update(obs)
    assert engine.states[1].session_id == session_id
    assert engine.states[1].sit_since == sit_since
    assert sink.closed == []


def test_away_grace_then_close():
    clock = FakeClock(datetime(2026, 1, 5, 9, 0, tzinfo=TZ))
    sink = TrackedSink(min_session_sec=60)
    engine = build_engine(clock, sink, enter_confirm_sec=5, leave_confirm_sec=10, away_grace_sec=60)
    obs = {1: Observation(desk_id=1, employee_id=7, employee_name="Budi", track_id=3)}

    for _ in range(3):
        engine.update(obs)
        clock.advance(10)
    engine.update(obs)
    session_id = engine.states[1].session_id
    assert session_id is not None

    clock.advance(11)
    events = engine.update({})
    assert [e.type for e in events] == ["AWAY_START"]

    clock.advance(60)
    events = engine.update({})
    assert [e.type for e in events] == ["SESSION_CLOSED"]
    assert events[0].duration_sec >= 60
    assert events[0].away_sec >= 60
    assert engine.states[1].session_id is None
    assert engine.states[1].status is Status.AWAY


def test_short_session_is_discarded():
    clock = FakeClock(datetime(2026, 1, 5, 9, 0, tzinfo=TZ))
    sink = TrackedSink(min_session_sec=600)
    engine = build_engine(clock, sink, enter_confirm_sec=1, leave_confirm_sec=1, away_grace_sec=5)
    obs = {1: Observation(desk_id=1, employee_id=7, employee_name="Budi", track_id=3)}

    engine.update(obs)
    clock.advance(2)
    engine.update(obs)
    clock.advance(2)
    engine.update({})
    clock.advance(10)
    events = engine.update({})
    assert [e.type for e in events] == ["SESSION_DISCARDED"]


def test_identity_can_change_mid_session():
    clock = FakeClock(datetime(2026, 1, 5, 9, 0, tzinfo=TZ))
    sink = TrackedSink()
    engine = build_engine(clock, sink, enter_confirm_sec=1)
    engine.update({1: Observation(desk_id=1, employee_id=7, employee_name="Budi", track_id=3)})
    clock.advance(5)
    events = engine.update({1: Observation(desk_id=1, employee_id=7, employee_name="Budi", track_id=3)})
    assert [e.type for e in events] == ["SESSION_OPENED"]
    session_id = engine.states[1].session_id
    assert session_id is not None

    clock.advance(5)
    events = engine.update({1: Observation(desk_id=1, employee_id=9, employee_name="Sari", track_id=3)})
    assert [e.type for e in events] == ["IDENTIFIED"]
    assert events[0].employee_id == 9
    assert sink.identities == [(session_id, 9)]
    assert engine.states[1].employee_name == "Sari"


def test_unidentified_person_still_counted():
    clock = FakeClock(datetime(2026, 1, 5, 9, 0, tzinfo=TZ))
    sink = TrackedSink()
    engine = build_engine(clock, sink, enter_confirm_sec=1)
    obs = {1: Observation(desk_id=1, track_id=3)}
    engine.update(obs)
    clock.advance(2)
    events = engine.update(obs)
    assert [e.type for e in events] == ["SESSION_OPENED"]
    assert events[0].employee_id is None
    assert engine.states[1].duration_sec(engine._clock()) >= 2


def test_desks_are_independent():
    clock = FakeClock(datetime(2026, 1, 5, 9, 0, tzinfo=TZ))
    sink = TrackedSink(min_session_sec=60)
    engine = build_engine(clock, sink, enter_confirm_sec=1, leave_confirm_sec=1, away_grace_sec=5)
    engine.update({1: Observation(desk_id=1, employee_id=7, track_id=3)})
    clock.advance(2)
    engine.update({1: Observation(desk_id=1, employee_id=7, track_id=3), 2: Observation(desk_id=2, employee_id=8, track_id=4)})
    assert engine.states[1].status is Status.PRESENT
    assert engine.states[2].status is Status.AWAY

    clock.advance(2)
    engine.update({2: Observation(desk_id=2, employee_id=8, track_id=4)})
    assert engine.states[1].status is Status.AWAY
    assert engine.states[2].status is Status.PRESENT

    clock.advance(4)
    events = engine.update({})
    closed = [e.desk_id for e in events if e.type in {"SESSION_CLOSED", "SESSION_DISCARDED"}]
    assert closed == [1]
    assert engine.states[1].session_id is None
    assert engine.states[2].session_id is not None


def test_flush_closes_open_session_on_stop():
    clock = FakeClock(datetime(2026, 1, 5, 9, 0, tzinfo=TZ))
    sink = TrackedSink(min_session_sec=60)
    engine = build_engine(clock, sink, enter_confirm_sec=1)
    engine.update({1: Observation(desk_id=1, employee_id=7, track_id=3)})
    clock.advance(30)
    engine.update({1: Observation(desk_id=1, employee_id=7, track_id=3)})
    clock.advance(60)
    engine.update({1: Observation(desk_id=1, employee_id=7, track_id=3)})
    events = engine.flush()
    assert [e.type for e in events] == ["SESSION_CLOSED"]
    assert events[0].duration_sec >= 90
    assert engine.states[1].session_id is None


def test_restore_open_session_after_restart():
    clock = FakeClock(datetime(2026, 1, 5, 9, 30, tzinfo=TZ))
    sink = TrackedSink()
    engine = build_engine(clock, sink)
    sit_start = datetime(2026, 1, 5, 9, 0, tzinfo=TZ)
    engine.restore_open_session(1, 55, 7, "Budi", sit_start, away_sec=120, away_count=2)
    state = engine.states[1]
    assert state.session_id == 55
    assert state.sit_since == sit_start
    assert state.away_total_sec == 120
    assert state.away_count == 2
    assert state.duration_sec(clock.now) == 1800
    assert sink.opened == []


def test_snapshot_shape():
    clock = FakeClock(datetime(2026, 1, 5, 9, 0, tzinfo=TZ))
    engine = build_engine(clock, TrackedSink())
    engine.update({1: Observation(desk_id=1, employee_id=7, employee_name="Budi", track_id=3)})
    clock.advance(60)
    rows = engine.snapshot()
    assert len(rows) == 2
    row = rows[0]
    assert row["label"] == "Kursi 1"
    assert row["status"] == "AWAY"
    assert row["employee_name"] == "Budi"
    assert row["duration_sec"] == 0


def test_return_from_away_resumes_present_session():
    clock = FakeClock(datetime(2026, 1, 5, 9, 0, tzinfo=TZ))
    sink = TrackedSink(min_session_sec=60)
    engine = build_engine(clock, sink, enter_confirm_sec=1, leave_confirm_sec=1, away_grace_sec=30)
    obs = {1: Observation(desk_id=1, employee_id=7, employee_name="Budi", track_id=3)}

    engine.update(obs)
    clock.advance(2)
    engine.update(obs)
    assert engine.states[1].status is Status.PRESENT
    session_id = engine.states[1].session_id

    clock.advance(2)
    engine.update({})
    assert engine.states[1].status is Status.AWAY

    clock.advance(2)
    events = engine.update(obs)
    assert [e.type for e in events] == ["AWAY_END"]
    assert engine.states[1].status is Status.PRESENT, "orang kembali -> status harus PRESENT lagi"
    assert engine.states[1].session_id == session_id, "sesi harus dilanjutkan, bukan ditutup"
    assert engine.states[1].away_total_sec >= 2
    assert engine.states[1].away_count == 1
    assert sink.closed == []


def test_flicker_does_not_close_open_session():
    clock = FakeClock(datetime(2026, 1, 5, 9, 0, tzinfo=TZ))
    sink = TrackedSink(min_session_sec=60)
    engine = build_engine(clock, sink, enter_confirm_sec=1, leave_confirm_sec=2, away_grace_sec=30)
    obs = {1: Observation(desk_id=1, employee_id=7, employee_name="Budi", track_id=3)}

    engine.update(obs)
    clock.advance(2)
    engine.update(obs)
    for _ in range(20):
        clock.advance(0.5)
        engine.update({} if _ % 2 == 0 else obs)
    assert sink.closed == [], "kedipan deteksi singkat tidak boleh menutup sesi"
    assert engine.states[1].status is Status.PRESENT


def test_short_detection_gap_does_not_reset_confirmation():
    clock = FakeClock(datetime(2026, 1, 5, 9, 0, tzinfo=TZ))
    sink = TrackedSink()
    engine = build_engine(clock, sink, enter_confirm_sec=8, leave_confirm_sec=4)
    obs = {1: Observation(desk_id=1, employee_id=7, employee_name="Budi", track_id=3)}

    engine.update(obs)
    clock.advance(2)
    engine.update({})
    clock.advance(2)
    engine.update(obs)
    clock.advance(2)
    engine.update({})
    clock.advance(3)
    events = engine.update(obs)
    assert [e.type for e in events] == ["SESSION_OPENED"], "flicker <= leave_confirm tidak boleh mereset timer"
    assert len(sink.opened) == 1
    assert engine.states[1].sit_since == datetime(2026, 1, 5, 9, 0, tzinfo=TZ)


def test_long_gap_resets_confirmation():
    clock = FakeClock(datetime(2026, 1, 5, 9, 0, tzinfo=TZ))
    sink = TrackedSink()
    engine = build_engine(clock, sink, enter_confirm_sec=8, leave_confirm_sec=3)
    obs = {1: Observation(desk_id=1, employee_id=7, employee_name="Budi", track_id=3)}

    engine.update(obs)
    clock.advance(6)
    engine.update({})
    clock.advance(5)
    engine.update({})
    assert sink.opened == []
    engine.update(obs)
    clock.advance(5)
    events = engine.update(obs)
    assert [e.type for e in events] == []
    clock.advance(4)
    events = engine.update(obs)
    assert [e.type for e in events] == ["SESSION_OPENED"]
    assert engine.states[1].sit_since == clock.now - timedelta(seconds=9)


@pytest.mark.parametrize("grace", [0, 1, 30])
def test_grace_values_are_accepted(grace: int):
    clock = FakeClock(datetime(2026, 1, 5, 9, 0, tzinfo=TZ))
    engine = build_engine(clock, TrackedSink(), enter_confirm_sec=grace, leave_confirm_sec=grace)
    assert engine.states[1].status is Status.AWAY