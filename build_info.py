"""Build and runtime metadata for the Read Aloud server."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

APP_VERSION = "v1.0.0-beta.1"
REPO_ROOT = Path(__file__).resolve().parent
SERVER_STARTED_AT = datetime.now(timezone.utc)


def _git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return ""


def git_commit() -> str:
    return _git_output("rev-parse", "--short", "HEAD") or "unknown"


def git_describe() -> str:
    return _git_output("describe", "--tags", "--always", "--dirty") or git_commit()


def git_commit_date() -> str | None:
    value = _git_output("log", "-1", "--format=%cI")
    return value or None


def app_js_version() -> str:
    app_js = REPO_ROOT / "static" / "app.js"
    if app_js.is_file():
        return str(int(app_js.stat().st_mtime))
    return "0"


def get_build_info() -> dict[str, str | bool | None]:
    describe = git_describe()
    return {
        "version": APP_VERSION,
        "git_describe": describe,
        "git_commit": git_commit(),
        "git_dirty": describe.endswith("-dirty"),
        "git_commit_date": git_commit_date(),
        "server_started_at": SERVER_STARTED_AT.isoformat(),
        "app_js_version": app_js_version(),
    }