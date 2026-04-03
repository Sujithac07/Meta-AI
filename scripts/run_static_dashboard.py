"""
Serve the static React dashboard HTML over HTTP (port 3000 by default).

Run from repo root:
  python scripts/run_static_dashboard.py
"""

from __future__ import annotations

import http.server
import os
import socketserver
import webbrowser
from threading import Timer

PORT = 3000
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIRECTORY = os.path.join(_ROOT, "frontend", "react-dashboard")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def log_message(self, format, *args):
        if args:
            print("[Dashboard] {}".format(args[0]))


def open_browser():
    webbrowser.open("http://localhost:{}".format(PORT))


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Static dashboard (frontend/react-dashboard)")
    print("=" * 60)
    print("\nURL: http://localhost:{}".format(PORT))
    print("Serving from: {}".format(DIRECTORY))
    print("\nPress Ctrl+C to stop\n")

    Timer(1, open_browser).start()

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nStopped.")
