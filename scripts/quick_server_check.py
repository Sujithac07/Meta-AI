"""Probe ports 3000/8000; optionally start static dashboard + backend from repo root."""
import os
import socket
import subprocess
import sys
import time

_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_STATIC = os.path.join(_BASE, "scripts", "run_static_dashboard.py")
_BACKEND = os.path.join(_BASE, "backend_api_main.py")


def _port_open(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    r = s.connect_ex(("localhost", port))
    s.close()
    return r == 0


if __name__ == "__main__":
    ok3000 = _port_open(3000)
    ok8000 = _port_open(8000)
    print(f"Port 3000: {'LISTENING' if ok3000 else 'NOT LISTENING'}")
    print(f"Port 8000: {'LISTENING' if ok8000 else 'NOT LISTENING'}")

    if not ok3000 or not ok8000:
        if not ok3000:
            print("Starting static dashboard...")
            subprocess.Popen([sys.executable, _STATIC], cwd=_BASE)
        if not ok8000:
            print("Starting backend_api_main.py...")
            subprocess.Popen([sys.executable, _BACKEND], cwd=_BASE)
        print("Waiting 5s...")
        time.sleep(5)
        ok3000 = _port_open(3000)
        ok8000 = _port_open(8000)
        print(f"Port 3000: {'LISTENING' if ok3000 else 'NOT LISTENING'}")
        print(f"Port 8000: {'LISTENING' if ok8000 else 'NOT LISTENING'}")
    else:
        print("Both ports already in use.")
