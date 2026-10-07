from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

from ..config import load_config
from .routes import router

PACKAGE_ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_ROOT.parents[2] / "web" / "templates"


def create_app() -> FastAPI:
    config = load_config()
    app = FastAPI(title="Employee Presence Monitor", version="1.0.0")
    app.include_router(router)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.globals["poll_seconds"] = config.server.poll_seconds
    templates.env.globals["timezone"] = config.timezone
    stream_host = config.server.stream_host
    if stream_host in {"0.0.0.0", "::"}:
        stream_host = "127.0.0.1"
    templates.env.globals["stream_url"] = f"http://{stream_host}:{config.server.stream_port}/stream"
    templates.env.globals["stream_fps"] = config.server.stream_fps
    templates.env.globals["stream_port"] = config.server.stream_port

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/")
    def index(request: Request):
        response = templates.TemplateResponse(request, "index.html")
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response

    return app


app = create_app()