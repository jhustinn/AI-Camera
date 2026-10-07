CREATE TABLE IF NOT EXISTS cameras (
    id          SERIAL PRIMARY KEY,
    camera_key  TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    source      TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS employees (
    id           SERIAL PRIMARY KEY,
    employee_no  TEXT UNIQUE,
    name         TEXT NOT NULL,
    dept         TEXT,
    embedding    BYTEA,
    embedding_dim INTEGER,
    face_count   INTEGER NOT NULL DEFAULT 0,
    photo_path   TEXT,
    active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS desks (
    id          SERIAL PRIMARY KEY,
    camera_id   INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    employee_id INTEGER REFERENCES employees(id) ON DELETE SET NULL,
    roi         JSONB NOT NULL DEFAULT '[0,0,1,1]'::jsonb,
    active      BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (camera_id, label)
);

CREATE INDEX IF NOT EXISTS ix_desks_employee ON desks (employee_id);

CREATE TABLE IF NOT EXISTS presence_sessions (
    id           BIGSERIAL PRIMARY KEY,
    desk_id      INTEGER NOT NULL REFERENCES desks(id) ON DELETE CASCADE,
    camera_id    INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    employee_id  INTEGER REFERENCES employees(id) ON DELETE SET NULL,
    track_id     INTEGER,
    status       TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','closed','discarded')),
    sit_start    TIMESTAMPTZ NOT NULL,
    sit_end      TIMESTAMPTZ,
    duration_sec INTEGER,
    away_sec     INTEGER NOT NULL DEFAULT 0,
    away_count   INTEGER NOT NULL DEFAULT 0,
    note         TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_sessions_employee_start
    ON presence_sessions (employee_id, sit_start DESC);
CREATE INDEX IF NOT EXISTS ix_sessions_desk_start
    ON presence_sessions (desk_id, sit_start DESC);
CREATE INDEX IF NOT EXISTS ix_sessions_open
    ON presence_sessions (desk_id) WHERE status = 'open';
CREATE UNIQUE INDEX IF NOT EXISTS ux_sessions_one_open_per_desk
    ON presence_sessions (desk_id) WHERE status = 'open';

CREATE TABLE IF NOT EXISTS session_events (
    id          BIGSERIAL PRIMARY KEY,
    session_id  BIGINT REFERENCES presence_sessions(id) ON DELETE CASCADE,
    desk_id     INTEGER,
    employee_id INTEGER,
    event_type  TEXT NOT NULL,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    note        TEXT
);

CREATE INDEX IF NOT EXISTS ix_events_ts ON session_events (ts DESC);

CREATE TABLE IF NOT EXISTS live_status (
    desk_id        INTEGER PRIMARY KEY REFERENCES desks(id) ON DELETE CASCADE,
    label          TEXT NOT NULL,
    status         TEXT NOT NULL CHECK (status IN ('PRESENT','AWAY','UNKNOWN')),
    employee_id    INTEGER,
    employee_name  TEXT,
    track_id       INTEGER,
    sit_since      TIMESTAMPTZ,
    duration_sec   INTEGER NOT NULL DEFAULT 0,
    away_sec       INTEGER NOT NULL DEFAULT 0,
    session_id     BIGINT,
    fps            REAL,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE OR REPLACE VIEW daily_summary AS
SELECT
    s.employee_id,
    e.name AS employee_name,
    (s.sit_start AT TIME ZONE 'UTC')::DATE AS work_date,
    SUM(COALESCE(s.duration_sec, 0))::BIGINT AS total_sit_sec,
    SUM(s.away_sec)::BIGINT AS total_away_sec,
    COUNT(*)::BIGINT AS session_count,
    MIN(s.sit_start) AS first_in,
    MAX(COALESCE(s.sit_end, s.sit_start)) AS last_out
FROM presence_sessions s
LEFT JOIN employees e ON e.id = s.employee_id
WHERE s.status <> 'discarded'
GROUP BY s.employee_id, e.name, (s.sit_start AT TIME ZONE 'UTC')::DATE;