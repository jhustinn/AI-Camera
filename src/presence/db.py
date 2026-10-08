from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

load_dotenv()
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/employee_presence"
)


_local = threading.local()


def _connect() -> psycopg.Connection:
    conn = getattr(_local, "conn", None)
    if conn is not None:
        if conn.closed:
            conn = None
        else:
            return conn
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    _local.conn = conn
    return conn


def close_connections() -> None:
    conn = getattr(_local, "conn", None)
    if conn is not None and not conn.closed:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
    _local.conn = None


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    conn = _connect()
    try:
        yield conn
        if conn.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            close_connections()
        raise


def get_conn_fresh() -> psycopg.Connection:
    return _connect()


def encode_embedding(embedding: np.ndarray) -> bytes:
    return np.asarray(embedding, dtype=np.float32).tobytes()


def decode_embedding(raw: bytes | memoryview | None) -> np.ndarray | None:
    if raw is None:
        return None
    return np.frombuffer(bytes(raw), dtype=np.float32)


def _rows(cur: psycopg.Cursor) -> list[dict[str, Any]]:
    return list(cur.fetchall())


def apply_schema(schema_path: str | Path | None = None) -> None:
    path = schema_path or (Path(__file__).resolve().parents[2] / "db" / "schema.sql")
    sql = Path(path).read_text(encoding="utf-8")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


def ensure_camera(camera_key: str, name: str, source: str) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO cameras (camera_key, name, source)
                VALUES (%s, %s, %s)
                ON CONFLICT (camera_key) DO UPDATE SET name = EXCLUDED.name, source = EXCLUDED.source
                RETURNING id
                """,
                (camera_key, name, str(source)),
            )
            camera_id = cur.fetchone()["id"]
        conn.commit()
    return int(camera_id)


def upsert_desk(camera_id: int, desk_id: int, label: str, roi: Sequence[float], employee_id: int | None = None) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO desks (camera_id, label, roi, employee_id)
                VALUES (%s, %s, %s::jsonb, %s)
                ON CONFLICT (camera_id, label) DO UPDATE
                SET roi = EXCLUDED.roi,
                    employee_id = COALESCE(EXCLUDED.employee_id, desks.employee_id)
                """,
                (camera_id, label, json.dumps([float(v) for v in roi]), employee_id),
            )
        conn.commit()


def sync_desks(camera_id: int, desks: Sequence[dict[str, Any]]) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            for desk in desks:
                cur.execute(
                    """
                    INSERT INTO desks (camera_id, label, roi, employee_id)
                    VALUES (%s, %s, %s::jsonb, %s)
                    ON CONFLICT (camera_id, label) DO UPDATE
                    SET roi = EXCLUDED.roi,
                        employee_id = COALESCE(EXCLUDED.employee_id, desks.employee_id)
                    """,
                    (camera_id, desk["label"], json.dumps([float(v) for v in desk["roi"]]), desk.get("employee_id")),
                )
        conn.commit()


def delete_employee(employee_id: int) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE presence_sessions SET employee_id = NULL
                WHERE employee_id = %s AND status = 'open'
                """,
                (employee_id,),
            )
            cur.execute("DELETE FROM employees WHERE id = %s", (employee_id,))
        conn.commit()


def list_desks(include_inactive: bool = False, camera_id: int | None = None) -> list[dict[str, Any]]:
    sql = """
        SELECT d.id, d.label, d.roi, d.active, d.camera_id,
               d.employee_id, e.name AS employee_name
        FROM desks d
        LEFT JOIN employees e ON e.id = d.employee_id
        WHERE (%s = TRUE OR d.active)
          AND (%s::int IS NULL OR d.camera_id = %s)
        ORDER BY d.id
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (include_inactive, camera_id, camera_id))
            return _rows(cur)


def list_desks_for_camera(camera_id: int) -> list[dict[str, Any]]:
    return list_desks(include_inactive=True, camera_id=camera_id)


def add_employee(name: str, employee_no: str | None, dept: str | None, photo_path: str | None) -> int:
    """Buat karyawan baru.

    employee_no bersifat opsional dan unik. Bila employee_no sudah dipakai:
    - nama sama (huruf besar/kecil diabaikan) -> pakai baris itu (enroll ulang)
    - nama berbeda -> tolak, supaya tidak menimpa karyawan lain diam-diam.
    """
    clean_no = (employee_no or "").strip() or None
    with get_conn() as conn:
        with conn.cursor() as cur:
            if clean_no is not None:
                cur.execute(
                    "SELECT id, name FROM employees WHERE employee_no = %s FOR UPDATE",
                    (clean_no,),
                )
                existing = cur.fetchone()
                if existing is not None:
                    if (existing["name"] or "").strip().lower() != name.strip().lower():
                        raise ValueError(
                            f"No. karyawan '{clean_no}' sudah dipakai oleh '{existing['name']}'. "
                            "Isi No. karyawan dengan nilai berbeda atau kosongkan."
                        )
                    cur.execute(
                        """
                        UPDATE employees SET name = %s, dept = COALESCE(%s, dept),
                            photo_path = COALESCE(%s, photo_path), updated_at = now()
                        WHERE id = %s RETURNING id
                        """,
                        (name, dept, photo_path, existing["id"]),
                    )
                    employee_id = int(cur.fetchone()["id"])
                    conn.commit()
                    return employee_id
            cur.execute(
                """
                INSERT INTO employees (name, employee_no, dept, photo_path)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (name, clean_no, dept, photo_path),
            )
            employee_id = int(cur.fetchone()["id"])
        conn.commit()
    return employee_id


def save_employee_embedding(employee_id: int, embedding: np.ndarray, face_count: int) -> None:
    vector = np.asarray(embedding, dtype=np.float32)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE employees
                SET embedding = %s, embedding_dim = %s, face_count = %s, updated_at = now()
                WHERE id = %s
                """,
                (encode_embedding(vector), int(vector.shape[0]), face_count, employee_id),
            )
        conn.commit()


def list_employees(only_with_embedding: bool = False, active_only: bool = True) -> list[dict[str, Any]]:
    sql = """
        SELECT id, name, employee_no, dept, face_count, active, created_at, updated_at,
               (embedding IS NOT NULL) AS has_embedding
        FROM employees
        WHERE (%s = FALSE OR embedding IS NOT NULL)
          AND (%s = FALSE OR active)
        ORDER BY name
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (only_with_embedding, active_only))
            return _rows(cur)


def load_known_faces() -> list[dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, embedding
                FROM employees
                WHERE active AND embedding IS NOT NULL
                """
            )
            rows = _rows(cur)
    known: list[dict[str, Any]] = []
    for row in rows:
        vector = decode_embedding(row["embedding"])
        if vector is not None and vector.size > 0:
            known.append({"id": int(row["id"]), "name": row["name"], "embedding": vector})
    return known


def open_session(
    desk_id: int,
    camera_id: int,
    employee_id: int | None,
    track_id: int | None,
    sit_start: datetime,
) -> int | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO presence_sessions (desk_id, camera_id, employee_id, track_id, sit_start)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING id
                """,
                (desk_id, camera_id, employee_id, track_id, sit_start),
            )
            row = cur.fetchone()
            session_id = int(row["id"]) if row else None
            if session_id is None:
                cur.execute(
                    "SELECT id FROM presence_sessions WHERE desk_id = %s AND status = 'open'",
                    (desk_id,),
                )
                existing = cur.fetchone()
                session_id = int(existing["id"]) if existing else None
        conn.commit()
    return session_id


def update_session_identity(session_id: int, employee_id: int) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE presence_sessions SET employee_id = %s
                WHERE id = %s AND status = 'open'
                """,
                (employee_id, session_id),
            )
        conn.commit()


def close_session(
    session_id: int,
    sit_end: datetime,
    away_sec: int,
    away_count: int,
    min_session_sec: int,
) -> tuple[str, int]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT sit_start FROM presence_sessions WHERE id = %s FOR UPDATE",
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                return "missing", 0
            duration = int((sit_end - row["sit_start"]).total_seconds())
            status = "closed" if duration >= min_session_sec else "discarded"
            cur.execute(
                """
                UPDATE presence_sessions
                SET sit_end = %s, duration_sec = %s, away_sec = %s, away_count = %s, status = %s
                WHERE id = %s
                """,
                (sit_end, duration, away_sec, away_count, status, session_id),
            )
        conn.commit()
    return status, duration


def get_open_session(desk_id: int) -> dict[str, Any] | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM presence_sessions WHERE desk_id = %s AND status = 'open'",
                (desk_id,),
            )
            row = cur.fetchone()
    return dict(row) if row else None


def record_event(
    event_type: str,
    desk_id: int | None = None,
    employee_id: int | None = None,
    session_id: int | None = None,
    note: str | None = None,
    ts: datetime | None = None,
) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO session_events (event_type, desk_id, employee_id, session_id, note, ts)
                VALUES (%s, %s, %s, %s, %s, COALESCE(%s, now()))
                """,
                (event_type, desk_id, employee_id, session_id, note, ts),
            )
        conn.commit()


def upsert_live_status(payload: dict[str, Any]) -> None:
    upsert_live_status_many([payload])


def upsert_live_status_many(rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    sql = """
        INSERT INTO live_status
            (desk_id, label, status, employee_id, employee_name, track_id,
             sit_since, duration_sec, away_sec, session_id, fps, updated_at)
        VALUES (%(desk_id)s, %(label)s, %(status)s, %(employee_id)s, %(employee_name)s,
                %(track_id)s, %(sit_since)s, %(duration_sec)s, %(away_sec)s,
                %(session_id)s, %(fps)s, now())
        ON CONFLICT (desk_id) DO UPDATE SET
            label = EXCLUDED.label,
            status = EXCLUDED.status,
            employee_id = EXCLUDED.employee_id,
            employee_name = EXCLUDED.employee_name,
            track_id = EXCLUDED.track_id,
            sit_since = EXCLUDED.sit_since,
            duration_sec = EXCLUDED.duration_sec,
            away_sec = EXCLUDED.away_sec,
            session_id = EXCLUDED.session_id,
            fps = EXCLUDED.fps,
            updated_at = now()
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, list(rows))
        conn.commit()


def clear_live_status() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM live_status")
        conn.commit()


def fetch_live_status() -> list[dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ls.*, d.employee_id AS assigned_employee_id
                FROM live_status ls
                JOIN desks d ON d.id = ls.desk_id
                ORDER BY ls.desk_id
                """
            )
            return _rows(cur)


def fetch_sessions(
    tz: str,
    day: str | None = None,
    employee_id: int | None = None,
    desk_id: int | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    sql = """
        SELECT s.id, s.desk_id, d.label AS desk_label, s.employee_id, e.name AS employee_name,
               s.status, s.sit_start, s.sit_end, s.duration_sec, s.away_sec, s.away_count,
               s.sit_start AT TIME ZONE %s AS sit_start_local,
               s.sit_end AT TIME ZONE %s AS sit_end_local
        FROM presence_sessions s
        JOIN desks d ON d.id = s.desk_id
        LEFT JOIN employees e ON e.id = s.employee_id
        WHERE (%s::date IS NULL OR (s.sit_start AT TIME ZONE %s)::date = %s::date)
          AND (%s::int IS NULL OR s.employee_id = %s)
          AND (%s::int IS NULL OR s.desk_id = %s)
        ORDER BY s.sit_start DESC
        LIMIT %s
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (tz, tz, day, tz, day, employee_id, employee_id, desk_id, desk_id, limit))
            return _rows(cur)


def fetch_daily_totals(tz: str, days: int = 7) -> list[dict[str, Any]]:
    sql = """
        WITH span AS (
            SELECT generate_series(
                (now() AT TIME ZONE %s)::date - (%s - 1),
                (now() AT TIME ZONE %s)::date,
                interval '1 day'
            )::date AS work_date
        )
        SELECT span.work_date,
               COALESCE(SUM(s.duration_sec), 0)::BIGINT AS total_sit_sec,
               COALESCE(SUM(s.away_sec), 0)::BIGINT AS total_away_sec,
               COUNT(s.id)::BIGINT AS session_count
        FROM span
        LEFT JOIN presence_sessions s
               ON (s.sit_start AT TIME ZONE %s)::date = span.work_date
              AND s.status <> 'discarded'
        GROUP BY span.work_date
        ORDER BY span.work_date
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (tz, days, tz, tz))
            return _rows(cur)


def fetch_employee_totals(tz: str, day: str | None = None) -> list[dict[str, Any]]:
    sql = """
        SELECT s.employee_id, e.name AS employee_name, e.dept,
               COALESCE(SUM(s.duration_sec), 0)::BIGINT AS total_sit_sec,
               COALESCE(SUM(s.away_sec), 0)::BIGINT AS total_away_sec,
               COUNT(s.id)::BIGINT AS session_count,
               MIN(s.sit_start AT TIME ZONE %s) AS first_in,
               MAX(COALESCE(s.sit_end, now()) AT TIME ZONE %s) AS last_out
        FROM presence_sessions s
        JOIN employees e ON e.id = s.employee_id
        WHERE s.status <> 'discarded'
          AND (%s::date IS NULL OR (s.sit_start AT TIME ZONE %s)::date = %s::date)
        GROUP BY s.employee_id, e.name, e.dept
        ORDER BY total_sit_sec DESC
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (tz, tz, day, tz, day))
            return _rows(cur)


def fetch_recent_events(limit: int = 40) -> list[dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ev.*, d.label AS desk_label, e.name AS employee_name
                FROM session_events ev
                LEFT JOIN desks d ON d.id = ev.desk_id
                LEFT JOIN employees e ON e.id = ev.employee_id
                ORDER BY ev.ts DESC
                LIMIT %s
                """,
                (limit,),
            )
            return _rows(cur)


def close_all_open_sessions(end_ts: datetime, note: str = "recorder stopped") -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, sit_start FROM presence_sessions WHERE status = 'open' FOR UPDATE"
            )
            rows = _rows(cur)
            for row in rows:
                duration = int((end_ts - row["sit_start"]).total_seconds())
                cur.execute(
                    """
                    UPDATE presence_sessions
                    SET sit_end = %s, duration_sec = %s, status = 'closed', note = %s
                    WHERE id = %s
                    """,
                    (end_ts, duration, note, row["id"]),
                )
        conn.commit()
    return len(rows)


def assign_employee_to_desk(desk_id: int, employee_id: int | None) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE desks SET employee_id = %s WHERE id = %s", (employee_id, desk_id))
        conn.commit()