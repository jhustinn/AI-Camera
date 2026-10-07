from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest

from presence import db


def _db_available() -> bool:
    try:
        db.apply_schema()
        db.list_desks()
        return True
    except Exception:  # noqa: BLE001
        return False


DB_AVAILABLE = _db_available()
pytestmark = pytest.mark.skipif(not DB_AVAILABLE, reason="PostgreSQL tidak tersedia (lihat .env)")


@pytest.fixture(scope="module")
def client():
    if not DB_AVAILABLE:
        pytest.skip("PostgreSQL tidak tersedia")
    from fastapi.testclient import TestClient

    from presence.web.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture(scope="module", autouse=True)
def seed_state():
    if not DB_AVAILABLE:
        return
    db.record_event("SELFTEST", note="tes api")
    yield


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_dashboard_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Monitoring Kehadiran" in response.text
    assert "canvas" in response.text


def test_status_endpoint(client):
    payload = client.get("/api/status").json()
    assert "desks" in payload
    assert payload["timezone"]


def test_desks_endpoint(client):
    desks = client.get("/api/desks").json()["desks"]
    assert len(desks) >= 1
    assert {"id", "label", "roi"} <= set(desks[0])


def test_employees_endpoint(client):
    employees = client.get("/api/employees").json()["employees"]
    assert isinstance(employees, list)
    for employee in employees:
        assert {"id", "name", "has_embedding"} <= set(employee)


def test_sessions_endpoint(client):
    payload = client.get("/api/sessions?limit=10").json()
    assert "sessions" in payload
    if payload["sessions"]:
        row = payload["sessions"][0]
        assert {"desk_label", "sit_start_local", "duration_sec"} <= set(row)


def test_sessions_filter_by_day(client):
    payload = client.get("/api/sessions?day=1999-01-01").json()
    assert payload["sessions"] == []


def test_daily_summary_endpoint(client):
    payload = client.get("/api/summary/daily?days=7").json()
    assert len(payload["daily"]) == 7
    assert {"work_date", "total_sit_sec", "total_away_sec"} <= set(payload["daily"][0])


def test_employee_summary_endpoint(client):
    payload = client.get("/api/summary/employees").json()
    assert isinstance(payload["employees"], list)


def test_events_endpoint(client):
    events = client.get("/api/events?limit=5").json()["events"]
    assert len(events) <= 5


def test_export_csv(client):
    response = client.get("/api/export.csv")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert response.text.splitlines()[0].startswith("id,employee_name,desk_label")


def test_assign_employee_to_desk(client):
    desks = client.get("/api/desks").json()["desks"]
    desk_id = desks[0]["id"]
    response = client.post("/api/assign", json={"desk_id": desk_id, "employee_id": None})
    assert response.status_code == 200
    updated = [d for d in client.get("/api/desks").json()["desks"] if d["id"] == desk_id]
    assert updated[0]["employee_id"] is None


def test_frame_endpoint_returns_404_when_missing(client):
    response = client.get("/api/frame")
    assert response.status_code in {200, 404}