from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from presence.config import _expand_env, load_config  # noqa: E402


def test_expand_env_replaces_value(monkeypatch):
    monkeypatch.setenv("CCTV_USER", "admin")
    assert _expand_env("rtsp://${CCTV_USER}@host") == "rtsp://admin@host"


def test_expand_env_supports_default(monkeypatch):
    monkeypatch.delenv("CCTV_MISSING", raising=False)
    assert _expand_env("${CCTV_MISSING:-admin}") == "admin"


def test_expand_env_keeps_placeholder_when_unset_and_no_default(monkeypatch):
    monkeypatch.delenv("CCTV_TOTALLY_MISSING", raising=False)
    # dibiarkan apa adanya supaya miskonfigurasi kelihatan saat runtime
    assert _expand_env("a${CCTV_TOTALLY_MISSING}b") == "a${CCTV_TOTALLY_MISSING}b"


def test_expand_env_prefers_env_over_default(monkeypatch):
    monkeypatch.setenv("CCTV_USER", "operator")
    assert _expand_env("${CCTV_USER:-admin}") == "operator"


def test_expand_env_ignores_bare_dollar():
    assert _expand_env("harga $5 dan ${x}") == "harga $5 dan ${x}"


def test_example_config_has_no_real_password():
    """Template yang masuk repo publik tidak boleh memuat kredensial asli."""
    text = (PROJECT_ROOT / "config.cctv.example.yaml").read_text(encoding="utf-8")
    assert "CCTV_PASSWORD" in text
    assert "${CCTV_PASSWORD}" in text
    # tidak ada kredensial literal di dalam URL
    assert "rtsp://admin:admin123@" not in text
    for line in text.splitlines():
        if "source:" in line and "rtsp://" in line:
            assert "${CCTV_PASSWORD}" in line, line


def test_real_config_uses_env_placeholders():
    real = PROJECT_ROOT / "config.cctv.yaml"
    if not real.exists():
        pytest.skip("config.cctv.yaml belum dibuat di mesin ini")
    text = real.read_text(encoding="utf-8")
    assert "rtsp://admin:admin123@" not in text, "kredensial tidak boleh hardcoded"


def test_gitignore_excludes_cctv_config():
    lines = {
        line.strip()
        for line in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert "config.cctv.yaml" in lines, "config berisi kredensial tidak boleh ter-commit"
    assert "config.cctv.example.yaml" not in lines, "template justru harus ikut ter-commit"


def test_cctv_config_loads_with_env_credentials(monkeypatch):
    monkeypatch.setenv("CCTV_USER", "admin")
    monkeypatch.setenv("CCTV_PASSWORD", "rahasia-uji")
    cfg = load_config(PROJECT_ROOT / "config.cctv.yaml")
    assert len(cfg.channels) == 4
    for channel in cfg.channels:
        assert channel.source.startswith("rtsp://admin:rahasia-uji@")
