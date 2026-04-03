from __future__ import annotations

from threading import Lock
from typing import Dict


class InMemoryTelemetry:
    """Simple process-level telemetry collector for API request monitoring."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._request_count = 0
        self._status_counts: Dict[str, int] = {}
        self._path_counts: Dict[str, int] = {}
        self._latencies_ms = []

    def record(self, path: str, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self._request_count += 1
            status_key = str(status_code)
            self._status_counts[status_key] = self._status_counts.get(status_key, 0) + 1
            self._path_counts[path] = self._path_counts.get(path, 0) + 1
            self._latencies_ms.append(float(latency_ms))

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            lat = sorted(self._latencies_ms)
            p95 = lat[int(0.95 * (len(lat) - 1))] if lat else 0.0
            avg = (sum(lat) / len(lat)) if lat else 0.0
            return {
                "request_count": self._request_count,
                "status_counts": dict(self._status_counts),
                "path_counts": dict(self._path_counts),
                "avg_latency_ms": round(avg, 2),
                "p95_latency_ms": round(p95, 2),
            }


telemetry = InMemoryTelemetry()
