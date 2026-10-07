from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest
import yaml

from presence.config import AppConfig, load_config

SECTIONS = {
    "camera": "camera",
    "detection": "detection",
    "face": "face",
    "presence": "presence",
    "pose": "pose",
    "server": "server",
}


@pytest.fixture(scope="module")
def raw_config() -> dict:
    text = (PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


@pytest.fixture(scope="module")
def config() -> AppConfig:
    return load_config(PROJECT_ROOT / "config.yaml")


@pytest.mark.parametrize("section", sorted(SECTIONS))
def test_every_yaml_key_is_loaded(section: str, raw_config: dict, config: AppConfig):
    values = raw_config.get(section, {})
    if not isinstance(values, dict):
        pytest.skip(f"bagian {section} bukan mapping")
    loaded = getattr(config, SECTIONS[section])
    known = {f.name for f in fields(loaded)}
    ignored = []
    for key, expected in values.items():
        if key not in known:
            ignored.append(key)
            continue
        actual = getattr(loaded, key)
        if isinstance(expected, list) and isinstance(actual, tuple):
            expected = tuple(expected)
        assert actual == expected, f"{section}.{key}: yaml={expected!r} tetapi loader={actual!r}"
    assert not ignored, f"key tidak dikenal di config: {ignored}"


def test_desks_are_parsed_with_roi(config: AppConfig):
    assert config.desks
    for desk in config.desks:
        assert len(desk.roi) == 4
        assert all(0.0 <= value <= 1.0 for value in desk.roi)


def test_source_resolution_handles_string_paths():
    from presence.config import _resolve_source

    assert _resolve_source(0) == 0
    assert _resolve_source("1") == 1
    assert _resolve_source("rtsp://10.0.0.1/stream") == "rtsp://10.0.0.1/stream"
    assert _resolve_source("D:/video/ke.mp4") == "D:/video/ke.mp4"


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config(PROJECT_ROOT / "config-tidak-ada.yaml")


def test_defaults_applied_for_missing_keys(tmp_path: Path):
    minimal = tmp_path / "config-min.yaml"
    minimal.write_text(
        "timezone: Asia/Jakarta\ncamera:\n  id: x\n  name: X\n  source: 0\npresence:\n  enter_confirm_sec: 3\n",
        encoding="utf-8",
    )
    cfg = load_config(minimal)
    assert cfg.presence.enter_confirm_sec == 3
    assert cfg.presence.away_grace_sec == 60
    assert cfg.pose.enabled is True
    assert cfg.server.stream_port == 8001