#!/usr/bin/env python3
"""
MetaAI Pro - Enterprise AutoML Platform
Professional Dashboard v3.0 with Advanced Data Ingestion
"""
import sys
import time
import os
import socket

start = time.time()


def _is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """Check whether a local TCP port is free for binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _pick_server_port(default_port: int = 7860, max_tries: int = 20) -> int:
    """Return an available port, preferring default_port and then nearby ports."""
    env_port = os.getenv("GRADIO_SERVER_PORT")
    if env_port:
        try:
            parsed = int(env_port)
            if _is_port_available(parsed):
                return parsed
        except ValueError:
            pass

    for port in range(default_port, default_port + max_tries):
        if _is_port_available(port):
            return port

    raise RuntimeError(
        f"No free port found in range {default_port}-{default_port + max_tries - 1}"
    )

try:
    print("\n" + "="*50)
    print("MetaAI Pro v3.0 - Enterprise AutoML")
    print("="*50)
    print("\n[MetaAI Pro] Loading modules...")
    
    from dashboard_v3 import build_dashboard
    
    print("[MetaAI Pro] Building dashboard...")
    
    app = build_dashboard()
    
    elapsed = time.time() - start
    server_port = _pick_server_port()
    print(f"[MetaAI Pro] Ready in {elapsed:.1f}s")
    print("\n" + "-"*50)
    print(f"Access: http://127.0.0.1:{server_port}")
    print("-"*50 + "\n")
    
    app.launch(
        server_name="127.0.0.1",
        server_port=server_port,
        show_error=True,
        share=False
    )

except KeyboardInterrupt:
    print("\n[MetaAI Pro] Stopped")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
