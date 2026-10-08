from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from presence.recorder import _is_file_source, _is_network_source  # noqa: E402

RTSP = "rtsp://admin:pwd@192.168.1.109:554/cam/realmonitor?channel=1&subtype=1"


def test_network_url_is_not_treated_as_file():
    """Regression: RTSP pernah dianggap file, jadi recorder berhenti permanen."""
    assert _is_network_source(RTSP) is True
    assert _is_file_source(RTSP) is False


def test_camera_index_is_neither_file_nor_network():
    assert _is_file_source("0") is False
    assert _is_file_source("1") is False
    assert _is_network_source("0") is False


def test_real_video_path_is_file():
    assert _is_file_source("data/seq/clip.mp4") is True
    assert _is_file_source(r"D:\video\monitoring.mp4") is True


def test_other_schemes_are_network():
    for url in ("rtsps://a/b", "http://a/b", "https://a/b", "rtmp://a/b"):
        assert _is_network_source(url) is True, url
        assert _is_file_source(url) is False, url


def test_non_string_and_empty():
    assert _is_file_source(None) is False
    assert _is_file_source(1) is False
    assert _is_file_source("   ") is False


def test_reconnect_keeps_trying_until_success():
    """Reconnect harus berulang, bukan menyerah setelah satu percobaan."""
    src = (PROJECT_ROOT / "src" / "presence" / "recorder.py").read_text(encoding="utf-8")
    body = src.split("def _reconnect", 1)[1].split("def _loop", 1)[0]
    assert "while not self._stop" in body, "_reconnect harus looping sampai link kembali"
    assert "stream tersambung kembali" in body
    # sleep harus bisa diinterupsi Ctrl+C
    assert "_sleep_interruptible" in body


def test_network_failover_threshold_is_low():
    """READ_TIMEOUT 10 dtk x 30 = 5 menit sebelum reconnect. Terlalu lama."""
    src = (PROJECT_ROOT / "src" / "presence" / "recorder.py").read_text(encoding="utf-8")
    assert "max_failures = 5 if _is_network_source(self._source) else 30" in src
