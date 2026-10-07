from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pytest

from presence.detector import Detection
from presence.pose import (
    BONES,
    KEYPOINT_NAMES,
    SITTING,
    STANDING,
    UNKNOWN,
    Skeleton,
    classify_posture,
    clip_to_bbox,
    match_skeletons,
    smooth_skeleton,
    validate_skeleton,
)


def make_keypoints(points: dict[int, tuple[float, float, float]] = None) -> np.ndarray:
    data = np.zeros((len(KEYPOINT_NAMES), 3), dtype=np.float32)
    for index, (x, y, conf) in (points or {}).items():
        data[index] = (x, y, conf)
    return data


def test_clip_to_bbox_removes_keypoints_outside_person():
    keypoints = make_keypoints(
        {
            0: (100, 100, 0.9),
            5: (110, 150, 0.9),
            6: (130, 150, 0.9),
            11: (115, 200, 0.9),
            12: (125, 200, 0.9),
            15: (900, 900, 0.9),
        }
    )
    skeleton = Skeleton(keypoints=keypoints, score=0.9)
    clipped, ratio = clip_to_bbox(skeleton, (80, 80, 200, 260), margin=0.05)
    assert clipped.keypoints[15][2] == 0.0
    assert clipped.keypoints[0][2] == pytest.approx(0.9)
    assert ratio == pytest.approx(5 / 6)
    assert skeleton.keypoints[15][2] == 0.9, "keypoints asli tidak boleh dimodifikasi"


def test_clip_to_bbox_ratio_is_zero_when_nothing_inside():
    keypoints = make_keypoints({0: (10, 10, 0.9), 5: (12, 12, 0.9)})
    clipped, ratio = clip_to_bbox(Skeleton(keypoints=keypoints, score=0.5), (500, 500, 600, 600))
    assert ratio == 0.0
    assert clipped.valid_count(0.3) == 0


def test_clip_to_bbox_respects_margin():
    keypoints = make_keypoints({0: (92, 100, 0.9)})
    skeleton = Skeleton(keypoints=keypoints, score=0.9)
    _, without_margin = clip_to_bbox(skeleton, (100, 80, 200, 300), margin=0.0)
    _, with_margin = clip_to_bbox(skeleton, (100, 80, 200, 300), margin=0.2)
    assert without_margin == 0.0
    assert with_margin == 1.0


def test_validate_skeleton_drops_implausible_leg():
    keypoints = make_keypoints(
        {
            5: (100, 100, 0.9),
            6: (140, 100, 0.9),
            11: (110, 200, 0.9),
            12: (130, 200, 0.9),
            13: (600, 700, 0.9),
            14: (130, 300, 0.9),
            15: (110, 400, 0.9),
            16: (130, 400, 0.9),
        }
    )
    validated = validate_skeleton(Skeleton(keypoints=keypoints, score=0.9))
    assert validated.keypoints[13][2] == 0.0, "lutut terlalu jauh dari pinggul harus dibuang"
    assert validated.keypoints[15][2] == 0.0
    assert validated.keypoints[14][2] > 0.0, "kaki yang wajar tetap dipakai"
    assert validated.keypoints[16][2] > 0.0


def test_validate_skeleton_keeps_anatomically_valid_body():
    keypoints = standing_front_view()
    validated = validate_skeleton(Skeleton(keypoints=keypoints, score=0.9))
    assert validated.valid_count(0.3) == int((keypoints[:, 2] >= 0.3).sum())


def test_validate_skeleton_does_not_mutate_input():
    keypoints = make_keypoints({5: (100, 100, 0.9), 6: (140, 100, 0.9), 11: (110, 200, 0.9), 12: (130, 200, 0.9), 13: (600, 700, 0.9)})
    skeleton = Skeleton(keypoints=keypoints, score=0.9)
    validate_skeleton(skeleton)
    assert skeleton.keypoints[13][2] == pytest.approx(0.9)


def test_smooth_skeleton_blends_previous_position():
    current = Skeleton(keypoints=make_keypoints({0: (110, 100, 0.9), 5: (120, 150, 0.9)}), score=0.9)
    previous = Skeleton(keypoints=make_keypoints({0: (100, 100, 0.9), 5: (120, 150, 0.9)}), score=0.9)
    smoothed = smooth_skeleton(current, previous, alpha=0.5)
    assert smoothed.keypoints[0][0] == pytest.approx(105.0)
    assert smoothed.keypoints[5][0] == pytest.approx(120.0)


def test_smooth_skeleton_keeps_current_when_no_previous():
    current = Skeleton(keypoints=make_keypoints({0: (110, 100, 0.9)}), score=0.9)
    smoothed = smooth_skeleton(current, None, alpha=0.35)
    assert smoothed.keypoints[0][0] == pytest.approx(110.0)


def test_smooth_skeleton_does_not_resurrect_dropped_keypoints():
    current = Skeleton(keypoints=make_keypoints({0: (110, 100, 0.0)}), score=0.9)
    previous = Skeleton(keypoints=make_keypoints({0: (100, 100, 0.9)}), score=0.9)
    smoothed = smooth_skeleton(current, previous, alpha=0.5)
    assert smoothed.keypoints[0][2] == 0.0
    assert smoothed.valid_count(0.3) == 0


def test_smooth_skeleton_alpha_one_returns_current():
    current = Skeleton(keypoints=make_keypoints({0: (110, 100, 0.9)}), score=0.9)
    previous = Skeleton(keypoints=make_keypoints({0: (100, 100, 0.9)}), score=0.9)
    smoothed = smooth_skeleton(current, previous, alpha=1.0)
    assert smoothed.keypoints[0][0] == pytest.approx(110.0)


def test_bones_reference_valid_keypoints():
    assert len(KEYPOINT_NAMES) == 17
    for start, end in BONES:
        assert 0 <= start < len(KEYPOINT_NAMES)
        assert 0 <= end < len(KEYPOINT_NAMES)
        assert start != end


def standing_side_view() -> np.ndarray:
    return make_keypoints(
        {
            0: (150, 60, 0.9),
            5: (140, 120, 0.9),
            6: (160, 120, 0.9),
            11: (145, 260, 0.9),
            12: (155, 260, 0.9),
            13: (145, 400, 0.85),
            14: (155, 400, 0.85),
            15: (145, 540, 0.8),
            16: (155, 540, 0.8),
        }
    )


def standing_front_view() -> np.ndarray:
    return make_keypoints(
        {
            0: (150, 60, 0.9),
            5: (110, 120, 0.9),
            6: (190, 120, 0.9),
            11: (115, 260, 0.9),
            12: (185, 260, 0.9),
            13: (115, 400, 0.85),
            14: (185, 400, 0.85),
            15: (115, 540, 0.8),
            16: (185, 540, 0.8),
        }
    )


def sitting_side_view() -> np.ndarray:
    return make_keypoints(
        {
            0: (150, 60, 0.9),
            5: (140, 120, 0.9),
            6: (158, 122, 0.9),
            11: (145, 255, 0.9),
            12: (152, 257, 0.9),
            13: (300, 250, 0.85),
            14: (302, 252, 0.85),
            15: (295, 400, 0.8),
            16: (297, 402, 0.8),
        }
    )


def sitting_front_view() -> np.ndarray:
    return make_keypoints(
        {
            0: (150, 70, 0.9),
            5: (105, 130, 0.9),
            6: (195, 130, 0.9),
            11: (112, 235, 0.9),
            12: (188, 235, 0.9),
            13: (100, 270, 0.85),
            14: (200, 270, 0.85),
            15: (100, 420, 0.8),
            16: (200, 420, 0.8),
        }
    )


def real_measured_standing_bus_jpg() -> np.ndarray:
    keypoints = make_keypoints(
        {
            0: (146, 420, 0.9),
            5: (95, 460, 0.9),
            6: (190, 455, 0.9),
            11: (120, 600, 0.9),
            12: (180, 600, 0.9),
            13: (125, 710, 0.9),
            14: (180, 710, 0.9),
            15: (135, 870, 0.85),
            16: (170, 870, 0.85),
        }
    )
    return keypoints


def real_measured_sitting_workspace() -> np.ndarray:
    return make_keypoints(
        {
            0: (456, 334, 0.9),
            5: (575, 379, 0.9),
            6: (574, 310, 0.9),
            11: (601, 560, 0.9),
            12: (607, 515, 0.9),
            13: (393, 608, 0.9),
            14: (420, 532, 0.9),
            15: (462, 720, 0.7),
            16: (462, 633, 0.6),
        }
    )


def test_posture_matches_real_standing_measurement():
    assert classify_posture(real_measured_standing_bus_jpg()) == STANDING


def test_posture_matches_real_sitting_measurement():
    assert classify_posture(real_measured_sitting_workspace()) == SITTING


@pytest.mark.parametrize(
    "builder,name",
    [(standing_side_view, "sisi"), (standing_front_view, "depan")],
)
def test_classify_posture_detects_standing(builder, name: str):
    assert classify_posture(builder()) == STANDING, f"gaya pandang {name}"


@pytest.mark.parametrize(
    "builder,name",
    [(sitting_side_view, "sisi"), (sitting_front_view, "depan")],
)
def test_classify_posture_detects_sitting(builder, name: str):
    assert classify_posture(builder()) == SITTING, f"gaya pandang {name}"


def test_side_view_sitting_is_not_mistaken_for_standing():
    keypoints = sitting_side_view()
    shoulders = np.linalg.norm(keypoints[5][:2] - keypoints[6][:2])
    torso = abs(float(keypoints[5][1] - keypoints[11][1]))
    assert shoulders < torso * 0.2, "leher bahu di view samping mepet, tidak bisa jadi skala"
    assert classify_posture(keypoints) == SITTING


def test_classify_posture_unknown_without_hips():
    keypoints = make_keypoints({5: (100, 100, 0.9), 6: (180, 100, 0.9)})
    assert classify_posture(keypoints) == UNKNOWN


def test_classify_posture_unknown_without_legs():
    keypoints = make_keypoints({5: (100, 100, 0.9), 6: (180, 100, 0.9), 11: (110, 250, 0.9), 12: (170, 250, 0.9)})
    assert classify_posture(keypoints) == UNKNOWN


def test_skeleton_posture_requires_enough_keypoints():
    skeleton = Skeleton(keypoints=standing_front_view(), score=0.9)
    assert skeleton.posture() == STANDING
    partial = Skeleton(
        keypoints=make_keypoints({5: (110, 120, 0.9), 6: (190, 120, 0.9)}),
        score=0.9,
    )
    assert partial.posture() == UNKNOWN


def test_classify_posture_unknown_when_confidence_low():
    keypoints = make_keypoints(
        {
            5: (100, 200, 0.9),
            6: (200, 200, 0.9),
            11: (110, 240, 0.1),
            12: (190, 240, 0.1),
        }
    )
    assert classify_posture(keypoints) == UNKNOWN


def test_skeleton_center_ignores_low_confidence_points():
    keypoints = make_keypoints({0: (50, 50, 0.9), 5: (70, 50, 0.9), 6: (90, 50, 0.9), 15: (900, 900, 0.05)})
    skeleton = Skeleton(keypoints=keypoints, score=0.9)
    assert skeleton.center() == pytest.approx((70.0, 50.0))
    assert skeleton.valid_count(0.3) == 3


def test_match_skeletons_pairs_nearest_detection():
    body_a = Skeleton(
        keypoints=make_keypoints(
            {0: (100, 100, 0.9), 5: (90, 150, 0.9), 6: (110, 150, 0.9), 11: (95, 200, 0.9), 12: (105, 200, 0.9)}
        ),
        score=0.9,
    )
    body_b = Skeleton(
        keypoints=make_keypoints(
            {0: (600, 100, 0.9), 5: (590, 150, 0.9), 6: (610, 150, 0.9), 11: (595, 200, 0.9), 12: (605, 200, 0.9)}
        ),
        score=0.9,
    )
    detections = [
        Detection(track_id=1, bbox=(60, 60, 160, 400), conf=0.9),
        Detection(track_id=2, bbox=(560, 60, 660, 400), conf=0.9),
    ]
    matched = match_skeletons([body_b, body_a], detections)
    assert set(matched) == {1, 2}
    assert matched[1] is body_a
    assert matched[2] is body_b


def test_match_skeletons_skips_far_away_skeletons():
    body = Skeleton(
        keypoints=make_keypoints(
            {0: (100, 100, 0.9), 5: (90, 150, 0.9), 6: (110, 150, 0.9), 11: (95, 200, 0.9), 12: (105, 200, 0.9)}
        ),
        score=0.9,
    )
    detections = [Detection(track_id=1, bbox=(900, 500, 1000, 900), conf=0.9)]
    assert match_skeletons([body], detections) == {}


def test_match_skeletons_ignores_untracked_detections():
    body = Skeleton(
        keypoints=make_keypoints(
            {0: (100, 100, 0.9), 5: (90, 150, 0.9), 6: (110, 150, 0.9), 11: (95, 200, 0.9), 12: (105, 200, 0.9)}
        ),
        score=0.9,
    )
    detections = [Detection(track_id=-1, bbox=(60, 60, 160, 400), conf=0.9)]
    assert match_skeletons([body], detections) == {}