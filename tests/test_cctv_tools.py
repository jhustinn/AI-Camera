from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pytest

import yaml

from rtsp_test import candidates, host_of
from watch_cctv import apply_to_config


def test_candidates_cover_common_nvr_brands():
    urls = [url for _label, url in candidates("192.168.1.14", "u", "p", 2)]
    joined = " ".join(urls)
    assert "Streaming/Channels/102" in joined
    assert "cam/realmonitor?channel=2" in joined
    assert "_channel=2_" in joined
    assert all(url.startswith("rtsp://u:p@192.168.1.14:554") for url in urls)


def test_candidates_encode_special_characters():
    urls = [url for _label, url in candidates("10.0.0.1", "rtsp user", "p@ss:word", 1, 8554)]
    assert urls[0].startswith("rtsp://rtsp%20user:p%40ss%3Aword@10.0.0.1:8554")


def test_host_of_parses_url():
    assert host_of("rtsp://user:pass@192.168.1.14:554/Streaming/Channels/101") == (
        "192.168.1.14",
        554,
    )


def test_apply_to_config_updates_matching_cameras(tmp_path: Path):
    config = tmp_path / "config.cctv.yaml"
    config.write_text(
        "timezone: Asia/Jakarta\n"
        "cameras:\n"
        "  - id: cctv-1\n"
        "    source: rtsp://u:p@192.168.1.14:554/Streaming/Channels/101\n"
        "    desks: []\n"
        "  - id: cctv-2\n"
        "    source: rtsp://u:p@192.168.1.14:554/Streaming/Channels/102\n"
        "    desks: []\n",
        encoding="utf-8",
    )
    found = {
        1: "rtsp://u:new@192.168.1.14:554/Streaming/Channels/201",
        2: "rtsp://u:new@192.168.1.14:554/Streaming/Channels/202",
    }
    changed = apply_to_config(config, found)
    assert changed == 2
    written = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert written["cameras"][0]["source"].endswith("201")
    assert written["cameras"][1]["source"].endswith("202")
    assert (tmp_path / "config.cctv.yaml.bak").exists(), "backup wajib dibuat"


def test_apply_to_config_ignores_unrelated_camera(tmp_path: Path):
    config = tmp_path / "config.cctv.yaml"
    config.write_text(
        "cameras:\n  - id: local\n    source: '0'\n    desks: []\n", encoding="utf-8"
    )
    assert apply_to_config(config, {1: "rtsp://x"}) == 0
    assert "source: '0'" in config.read_text(encoding="utf-8")


def test_apply_to_config_no_changes_without_findings(tmp_path: Path):
    config = tmp_path / "config.cctv.yaml"
    config.write_text("cameras:\n  - id: a\n    source: '0'\n", encoding="utf-8")
    before = config.read_text(encoding="utf-8")
    assert apply_to_config(config, {}) == 0
    assert config.read_text(encoding="utf-8") == before


def test_apply_to_config_handles_dahua_pattern(tmp_path: Path):
    config = tmp_path / "config.cctv.yaml"
    config.write_text(
        "cameras:\n"
        "  - id: cam1\n    source: rtsp://u:p@10.0.0.1:554/cam/realmonitor?channel=1&subtype=0\n",
        encoding="utf-8",
    )
    changed = apply_to_config(config, {1: "rtsp://u:new@10.0.0.1:554/cam/realmonitor?channel=1&subtype=1"})
    assert changed == 1
    assert "subtype=1" in config.read_text(encoding="utf-8")