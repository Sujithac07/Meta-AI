#!/usr/bin/env python3
"""Check ports and start static dashboard + FastAPI backend if needed."""
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_STATIC = os.path.join(_BASE, "scripts", "run_static_dashboard.py")
_BACKEND = os.path.join(_BASE, "backend_api_main.py")


def check_port(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    result = s.connect_ex(("localhost", port))
    s.close()
    return result == 0


def test_http(url: str, timeout: float = 5):
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="ignore")[:200]
    except urllib.error.HTTPError as e:
        return e.code, str(e)
    except urllib.error.URLError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


def main() -> int:
    print("=" * 60)
    print("SERVER STATUS CHECK")
    print("=" * 60)

    print("\n[1] Checking ports...")
    port_3000_open = check_port(3000)
    port_8000_open = check_port(8000)
    print(f"  Port 3000: {'OPEN' if port_3000_open else 'CLOSED'}")
    print(f"  Port 8000: {'OPEN' if port_8000_open else 'CLOSED'}")

    processes = []

    if not port_3000_open:
        print("\n[2a] Starting static dashboard (port 3000)...")
        if os.path.exists(_STATIC):
            p = subprocess.Popen(
                [sys.executable, _STATIC],
                cwd=_BASE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
            )
            processes.append(("dashboard", p))
            print(f"  Started with PID: {p.pid}")
        else:
            print(f"  ERROR: {_STATIC} not found!")
    else:
        print("\n[2a] Port 3000 already open, skipping")

    if not port_8000_open:
        print("\n[2b] Starting backend_api_main.py (port 8000)...")
        if os.path.exists(_BACKEND):
            p = subprocess.Popen(
                [sys.executable, _BACKEND],
                cwd=_BASE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
            )
            processes.append(("backend", p))
            print(f"  Started with PID: {p.pid}")
        else:
            print(f"  ERROR: {_BACKEND} not found!")
    else:
        print("\n[2b] Port 8000 already open, skipping")

    if processes:
        print("\n[3] Waiting 5 seconds for servers...")
        time.sleep(5)

    print("\n[4] Testing HTTP...")
    print("\n  http://localhost:3000 ...")
    status_3000, body_3000 = test_http("http://localhost:3000")
    if status_3000:
        print(f"  HTTP {status_3000}  preview: {body_3000[:100]}...")
    else:
        print(f"  FAILED: {body_3000}")

    print("\n  http://localhost:8000/health ...")
    status_8000, body_8000 = test_http("http://localhost:8000/health")
    if status_8000:
        print(f"  HTTP {status_8000}  {body_8000}")
    else:
        print(f"  FAILED: {body_8000}")

    print("\n" + "=" * 60)
    port_3000_final = check_port(3000)
    port_8000_final = check_port(8000)
    dashboard_ok = port_3000_final and status_3000
    backend_ok = port_8000_final and status_8000
    print(f"  Dashboard (3000): {'OK' if dashboard_ok else 'FAIL'}")
    print(f"  Backend (8000):   {'OK' if backend_ok else 'FAIL'}")
    print("=" * 60)

    return 0 if (dashboard_ok and backend_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
