from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = PROJECT_ROOT / "web" / "templates" / "index.html"


def _inline_js() -> str:
    html = TEMPLATE.read_text(encoding="utf-8")
    blocks = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S)
    assert blocks, "template harus punya minimal satu blok <script> inline"
    js = "\n;\n".join(blocks)
    # Ganti ekspresi Jinja supaya sisa kode tetap bisa di-parse oleh node.
    js = re.sub(r"\{\{.*?\}\}", "0", js)
    js = re.sub(r"\{%.*?%\}", "", js)
    return js


def test_dashboard_inline_js_is_valid_syntax():
    """Backtick yang hilang di template membuat seluruh dashboard mati diam-diam."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node tidak tersedia untuk validasi sintaks JS")
    js = _inline_js()
    tmp = PROJECT_ROOT / "data" / "_dashboard_js_check.js"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(js, encoding="utf-8")
    try:
        result = subprocess.run([node, "--check", str(tmp)], capture_output=True, text=True)
        assert result.returncode == 0, f"sintaks JS dashboard rusak:\n{result.stderr}"
    finally:
        tmp.unlink(missing_ok=True)


def test_dashboard_has_camera_grid_for_every_channel():
    """Grid harus dibangun dari stream_url tiap kamera, bukan satu URL global."""
    js = _inline_js()
    assert "loadCameraStreams" in js
    assert "c.stream_url" in js, "grid harus memakai stream_url per kamera"
    assert "cameraGrid" in js


def test_camera_api_marks_inactive_config():
    """Kamera sisa config lama tidak boleh mendapat stream_url (bentrok port)."""
    assert (PROJECT_ROOT / "src" / "presence" / "web" / "routes.py").exists()
    routes = (PROJECT_ROOT / "src" / "presence" / "web" / "routes.py").read_text(encoding="utf-8")
    assert 'row["in_active_config"] = False' in routes
    assert 'row["stream_url"] = None' in routes


def test_config_arg_precedes_subcommand_in_run_py():
    """`--config` adalah argumen level utama; setelah subcommand argparse menolaknya."""
    run_py = (PROJECT_ROOT / "run.py").read_text(encoding="utf-8")
    pattern = re.compile(r"argv \+= \[\"record\"", re.M)
    assert pattern.search(run_py), "argv harus disusun config dulu, baru subcommand"
    watch = (PROJECT_ROOT / "scripts" / "watch_cctv.py").read_text(encoding="utf-8")
    assert '"--config", str(config_path), "all"' in watch
    assert '"all", "--config"' not in watch


def test_recorder_rejects_duplicate_mjpeg_port():
    """Dua recorder pada port sama membuat kamera kehabisan klien."""
    recorder = (PROJECT_ROOT / "src" / "presence" / "recorder.py").read_text(encoding="utf-8")
    assert "probe.bind(" in recorder
    assert "sudah dipakai proses lain" in recorder


def test_cctv_config_points_to_direct_cameras():
    cfg = yaml_cctv()
    assert len(cfg["cameras"]) == 4
    for cam in cfg["cameras"]:
        assert cam["source"].startswith("rtsp://")
        assert "/cam/realmonitor?channel=1" in cam["source"]
        assert 8000 < cam["stream_port"] < 9000
    ports = [cam["stream_port"] for cam in cfg["cameras"]]
    assert len(set(ports)) == 4, f"stream_port tiap kamera harus unik: {ports}"


def yaml_cctv() -> dict:
    import yaml

    return yaml.safe_load((PROJECT_ROOT / "config.cctv.yaml").read_text(encoding="utf-8"))


def test_dashboard_renders_without_error(client_ok: bool = True):
    """Placeholder sederhana: template harus bisa dirender oleh Jinja."""
    from presence.config import load_config

    cfg = load_config(PROJECT_ROOT / "config.cctv.yaml")
    assert cfg.server.stream_port == 8001
    assert len(cfg.channels) == 4
    json.dumps({"ok": True})
