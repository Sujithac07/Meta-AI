"""Start FastAPI backend (8000) and static dashboard (3000) in separate consoles (Windows)."""
from __future__ import annotations

import os
import subprocess
import sys
import time

_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_STATIC = os.path.join(_BASE, "scripts", "run_static_dashboard.py")
_BACKEND = os.path.join(_BASE, "backend_api_main.py")

if __name__ == "__main__":
    flags = subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
    print("Starting backend (8000)...")
    subprocess.Popen([sys.executable, _BACKEND], cwd=_BASE, creationflags=flags)
    time.sleep(2)
    print("Starting static dashboard (3000)...")
    subprocess.Popen([sys.executable, _STATIC], cwd=_BASE, creationflags=flags)
    print("Dashboard: http://localhost:3000")
    print("API:       http://localhost:8000")
