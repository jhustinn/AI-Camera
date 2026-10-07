from __future__ import annotations

import csv
import io
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from ..config import load_config
from .. import db

router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG = load_config()
TZ = CONFIG.timezone


def _tz() -> str:
    return CONFIG.timezone


@router.get("/api/status")
def get_status() -> dict[str, Any]:
    rows = db.fetch_live_status()
    recorder_alive = bool(rows)
    return {
        "server_time": date.today().isoformat(),
        "timezone": _tz(),
        "recorder_running": recorder_alive,
        "desks": rows,
    }


@router.get("/api/desks")
def get_desks() -> dict[str, Any]:
    return {"desks": db.list_desks(), "timezone": _tz()}


@router.get("/api/employees")
def get_employees() -> dict[str, Any]:
    return {"employees": db.list_employees(), "timezone": _tz()}


@router.get("/api/sessions")
def get_sessions(
    day: str | None = Query(default=None),
    employee_id: int | None = Query(default=None),
    desk_id: int | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict[str, Any]:
    rows = db.fetch_sessions(_tz(), day=day, employee_id=employee_id, desk_id=desk_id, limit=limit)
    return {"sessions": rows, "day": day, "timezone": _tz()}


@router.get("/api/summary/daily")
def get_daily(days: int = Query(default=7, ge=1, le=90)) -> dict[str, Any]:
    return {"daily": db.fetch_daily_totals(_tz(), days=days), "timezone": _tz()}


@router.get("/api/summary/employees")
def get_employee_summary(day: str | None = Query(default=None)) -> dict[str, Any]:
    return {"employees": db.fetch_employee_totals(_tz(), day=day), "day": day, "timezone": _tz()}


@router.get("/api/events")
def get_events(limit: int = Query(default=40, ge=1, le=500)) -> dict[str, Any]:
    return {"events": db.fetch_recent_events(limit=limit), "timezone": _tz()}


@router.get("/api/export.csv")
def export_csv(
    day: str | None = Query(default=None),
    employee_id: int | None = Query(default=None),
) -> Response:
    rows = db.fetch_sessions(_tz(), day=day, employee_id=employee_id, limit=100000)
    buffer = io.StringIO()
    columns = [
        "id",
        "employee_name",
        "desk_label",
        "status",
        "sit_start_local",
        "sit_end_local",
        "duration_sec",
        "away_sec",
        "away_count",
    ]
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _stringify(row.get(key)) for key in columns})
    filename = f"presence-{day or 'all'}.csv"
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


class AssignRequest(BaseModel):
    desk_id: int
    employee_id: int | None = None


@router.post("/api/assign")
def assign(payload: AssignRequest) -> dict[str, Any]:
    try:
        db.assign_employee_to_desk(payload.desk_id, payload.employee_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _signal_reload()
    return {"ok": True, "desk_id": payload.desk_id, "employee_id": payload.employee_id}


@router.post("/api/reload-faces")
def reload_faces() -> dict[str, Any]:
    _signal_reload()
    return {"ok": True}


def _recorder_base() -> str:
    return f"http://{CONFIG.server.stream_host}:{CONFIG.server.stream_port}"


def _proxy(method: str, path: str, **kwargs: Any) -> Any:
    import httpx
    from fastapi import HTTPException

    try:
        response = httpx.request(method, f"{_recorder_base()}{path}", timeout=10.0, **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"recorder tidak merespons: {exc}") from exc
    if response.status_code >= 400:
        detail: Any
        try:
            detail = response.json().get("detail", response.text)
        except Exception:  # noqa: BLE001
            detail = response.text
        raise HTTPException(status_code=response.status_code, detail=detail)
    return response.json()


class EnrollStartRequest(BaseModel):
    name: str
    employee_no: str | None = None
    dept: str | None = None
    count: int = 10


@router.get("/api/enroll/status")
def enroll_status() -> dict[str, Any]:
    return _proxy("GET", "/enroll/status")


@router.post("/api/enroll/start")
def enroll_start(payload: EnrollStartRequest) -> dict[str, Any]:
    return _proxy("POST", "/enroll/start", json=payload.model_dump())


@router.post("/api/enroll/commit")
def enroll_commit() -> dict[str, Any]:
    return _proxy("POST", "/enroll/commit")


@router.post("/api/enroll/cancel")
def enroll_cancel() -> dict[str, Any]:
    return _proxy("POST", "/enroll/cancel")


@router.delete("/api/employees/{employee_id}")
def delete_employee(employee_id: int) -> dict[str, Any]:
    db.delete_employee(employee_id)
    (PROJECT_ROOT / "data" / "reload_faces.flag").touch()
    return {"ok": True, "deleted": employee_id}


@router.get("/stream")
def proxy_stream():
    from fastapi.responses import StreamingResponse

    base = _recorder_base()

    async def generator():
        import httpx

        try:
            async with httpx.AsyncClient(timeout=1.5) as probe:
                await probe.get(f"{base}/health")
        except Exception:  # noqa: BLE001
            return
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", f"{base}/stream") as response:
                    async for chunk in response.aiter_bytes(chunk_size=16384):
                        yield chunk
        except Exception:  # noqa: BLE001
            return

    return StreamingResponse(
        generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


@router.get("/api/frame")
def last_frame() -> FileResponse:
    path = PROJECT_ROOT / "data" / "last_frame.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="belum ada frame tersimpan")
    return FileResponse(path, media_type="image/jpeg")


def _signal_reload() -> None:
    flag = PROJECT_ROOT / "data" / "reload_faces.flag"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.touch()


def _stringify(value: Any) -> Any:
    if value is None:
        return ""
    return value