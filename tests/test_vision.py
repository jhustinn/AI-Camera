from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from presence.detector import DeskAssigner, Detection, assign_to_desks, face_region
from presence.face_id import FaceMatch, TrackIdentityRegistry

DESKS = [
    (1, "Kursi 1", (0.0, 0.0, 0.5, 1.0)),
    (2, "Kursi 2", (0.5, 0.0, 1.0, 1.0)),
]

DESKS_WITH_GAP = [
    (1, "Kursi 1", (0.0, 0.0, 0.4, 1.0)),
    (2, "Kursi 2", (0.6, 0.0, 1.0, 1.0)),
]


def test_detection_inside_left_desk():
    detections = [Detection(track_id=1, bbox=(10, 40, 200, 500), conf=0.9)]
    assigned = assign_to_desks(detections, DESKS, (720, 1280, 3), min_ratio=0.5)
    assert list(assigned) == [1]


def test_detection_inside_right_desk():
    detections = [Detection(track_id=1, bbox=(700, 40, 1200, 500), conf=0.9)]
    assigned = assign_to_desks(detections, DESKS, (720, 1280, 3), min_ratio=0.5)
    assert list(assigned) == [2]


def test_two_people_two_desks():
    detections = [
        Detection(track_id=1, bbox=(10, 40, 200, 500), conf=0.9),
        Detection(track_id=2, bbox=(700, 40, 1200, 500), conf=0.9),
    ]
    assigned = assign_to_desks(detections, DESKS, (720, 1280, 3), min_ratio=0.5)
    assert sorted(assigned) == [1, 2]


def test_person_in_gap_between_desks_is_ignored():
    detections = [Detection(track_id=1, bbox=(500, 40, 700, 500), conf=0.9)]
    assigned = assign_to_desks(detections, DESKS_WITH_GAP, (720, 1280, 3), min_ratio=0.5)
    assert assigned == {}


def test_center_inside_desk_wins_even_with_strict_ratio():
    detections = [Detection(track_id=1, bbox=(500, 100, 760, 700), conf=0.8)]
    assigned = assign_to_desks(detections, DESKS, (720, 1280, 3), min_ratio=0.9)
    assert list(assigned) == [1]


def test_assignner_keeps_previous_desk_while_still_overlapping():
    assigner = DeskAssigner(switch_grace_sec=8)
    desks = [(1, "Kursi 1", (0.0, 0.0, 0.42, 1.0)), (2, "Kursi 2", (0.42, 0.0, 1.0, 1.0))]
    inside_left = Detection(track_id=1, bbox=(100, 100, 500, 700), conf=0.9)
    assigner.assign([inside_left], desks, (720, 1280, 3), 0.4, now=0.0)
    assert assigner.desk_of(1) == 1

    straddling = Detection(track_id=1, bbox=(300, 100, 700, 700), conf=0.9)
    assigned = assigner.assign([straddling], desks, (720, 1280, 3), 0.4, now=1.0)
    assert list(assigned) == [1], "orang nyaring di batas jangan sampai pindah kursi"


def test_assignner_switches_after_grace_expires():
    assigner = DeskAssigner(switch_grace_sec=5)
    desks = [(1, "Kursi 1", (0.0, 0.0, 0.42, 1.0)), (2, "Kursi 2", (0.42, 0.0, 1.0, 1.0))]
    assigner.assign([Detection(track_id=1, bbox=(100, 100, 500, 700), conf=0.9)], desks, (720, 1280, 3), 0.4, now=0.0)
    moved = Detection(track_id=1, bbox=(900, 100, 1200, 700), conf=0.9)

    assigned = assigner.assign([moved], desks, (720, 1280, 3), 0.4, now=2.0)
    assert list(assigned) == [1], "masih dalam masa tenggang, tetap di kursi lama"

    assigned = assigner.assign([moved], desks, (720, 1280, 3), 0.4, now=6.0)
    assert list(assigned) == [2]
    assert assigner.desk_of(1) == 2


def test_assignner_forgets_dead_tracks():
    assigner = DeskAssigner(switch_grace_sec=5)
    desks = [(1, "Kursi 1", (0.0, 0.0, 1.0, 1.0))]
    assigner.assign([Detection(track_id=7, bbox=(10, 10, 400, 700), conf=0.9)], desks, (720, 1280, 3), 0.4, now=0.0)
    assert assigner.desk_of(7) == 1
    assigner.assign([], desks, (720, 1280, 3), 0.4, now=1.0)
    assert assigner.desk_of(7) is None


def test_assignner_ignores_untracked_detections():
    assigner = DeskAssigner()
    desks = [(1, "Kursi 1", (0.0, 0.0, 1.0, 1.0))]
    assigned = assigner.assign([Detection(track_id=-1, bbox=(10, 10, 400, 700), conf=0.9)], desks, (720, 1280, 3), 0.4, now=0.0)
    assert assigned == {}


def test_candidate_desks_returns_overlapping_desks():
    from presence.detector import candidate_desks

    desks = [(1, "Kursi 1", (0.0, 0.0, 0.5, 1.0)), (2, "Kursi 2", (0.5, 0.0, 1.0, 1.0))]
    detection = Detection(track_id=1, bbox=(600, 100, 760, 700), conf=0.9)
    ids = {desk_id for desk_id, _score, _center in candidate_desks(detection, desks, (720, 1280, 3), 0.4)}
    assert 2 in ids


def test_face_region_covers_upper_body():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    crop = face_region(frame, (400, 100, 600, 500), region_ratio=0.85, top_offset=0.2, widen=0.12)
    assert crop is not None
    assert crop.shape[0] == int(400 * 0.85)
    assert crop.shape[1] == pytest.approx(200 * 1.24, rel=0.05)


def test_face_region_rejects_tiny_box():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert face_region(frame, (10, 10, 40, 40), min_face_px=60) is None


def test_face_region_is_downscaled_when_too_wide():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    crop = face_region(frame, (10, 10, 1100, 700), max_width=320)
    assert crop is not None
    assert crop.shape[1] == 320


def test_one_person_never_occupies_two_desks():
    detections = [Detection(track_id=1, bbox=(600, 100, 800, 700), conf=0.9)]
    assigned = assign_to_desks(detections, DESKS, (720, 1280, 3), min_ratio=0.5)
    assert len(assigned) <= 1


def test_straddling_person_goes_to_desk_with_its_center():
    desks = [
        (1, "Kursi 1", (0.0, 0.0, 0.5, 1.0)),
        (2, "Kursi 2", (0.5, 0.0, 1.0, 1.0)),
    ]
    detections = [Detection(track_id=1, bbox=(560, 100, 900, 700), conf=0.9)]
    assigned = assign_to_desks(detections, desks, (720, 1280, 3), min_ratio=0.9)
    assert list(assigned) == [2]


def test_registry_requires_votes_before_locking():
    registry = TrackIdentityRegistry(votes_required=3)
    match = FaceMatch(7, "Budi", 0.8)
    for _ in range(2):
        employee_id, name, _sim = registry.observe(1, match)
        assert employee_id == 7
    employee_id, name, _sim = registry.observe(1, match)
    assert employee_id == 7
    assert registry.identity(1) == (7, "Budi")


def test_registry_ignores_low_similarity_match():
    registry = TrackIdentityRegistry(votes_required=2)
    low = FaceMatch(None, "Tidak dikenal", 0.1)
    for _ in range(5):
        assert registry.observe(2, low)[:2] == (None, None)
    assert registry.identity(2) == (None, None)


def test_registry_keeps_last_identity_after_releasing_track():
    registry = TrackIdentityRegistry(votes_required=1)
    registry.observe(3, FaceMatch(9, "Sari", 0.75))
    assert registry.release(3) == (9, "Sari")
    assert registry.identity(3) == (None, None)


def test_registry_prunes_dead_tracks():
    registry = TrackIdentityRegistry(votes_required=1)
    registry.observe(4, FaceMatch(9, "Sari", 0.75))
    registry.prune({5})
    assert registry.identity(4) == (None, None)


def test_anchor_inside_overrides_wide_straddling_bounding_box():
    # Kursi 1: x from 0.03 to 0.42 (38.4px to 537.6px)
    # Kursi 2: x from 0.42 to 0.85 (537.6px to 1088px)
    desks = [
        (1, "Kursi 1", (0.03, 0.10, 0.42, 1.0)),
        (2, "Kursi 2", (0.42, 0.10, 0.85, 1.0)),
    ]
    # Person with arm extending left (x=50 to x=1050)
    det = Detection(track_id=128, bbox=(50, 100, 1050, 700), conf=0.9)
    # Torso/skeleton anchor is firmly in Kursi 2 (x=700, y=350)
    anchor = (700.0, 350.0)

    assigner = DeskAssigner(switch_grace_sec=3)
    assigned = assigner.assign([det], desks, (720, 1280, 3), 0.4, now=0.0, anchors={128: anchor})
    assert list(assigned) == [2], "Meskipun bounding box melebar ke kiri, anchor torso di Kursi 2 harus membuat orang ter-assign ke Kursi 2"


def test_desk_assigner_switches_when_anchor_moves_to_another_desk():
    desks = [
        (1, "Kursi 1", (0.03, 0.10, 0.42, 1.0)),
        (2, "Kursi 2", (0.42, 0.10, 0.85, 1.0)),
    ]
    assigner = DeskAssigner(switch_grace_sec=3)
    # Initially in Kursi 1
    det1 = Detection(track_id=128, bbox=(100, 100, 450, 700), conf=0.9)
    anchor1 = (250.0, 350.0)
    assigned = assigner.assign([det1], desks, (720, 1280, 3), 0.4, now=0.0, anchors={128: anchor1})
    assert list(assigned) == [1]

    # Moves to Kursi 2 with arm still overlapping Kursi 1
    det2 = Detection(track_id=128, bbox=(50, 100, 1050, 700), conf=0.9)
    anchor2 = (700.0, 350.0)

    # Frame at t=0.2 (pending confirmation window)
    assigner.assign([det2], desks, (720, 1280, 3), 0.4, now=0.2, anchors={128: anchor2})

    # Frame at t=1.2 (sustained in Kursi 2 past confirm_time 0.8s)
    assigned_switched = assigner.assign([det2], desks, (720, 1280, 3), 0.4, now=1.2, anchors={128: anchor2})
    assert list(assigned_switched) == [2], "Setelah konfirmasi singkat, harus beralih ke Kursi 2"
    assert assigner.desk_of(128) == 2