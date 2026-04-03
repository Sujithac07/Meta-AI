import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

telemetry_path = ROOT / "app" / "utils" / "telemetry.py"
spec = importlib.util.spec_from_file_location("telemetry_mod", telemetry_path)
telemetry_mod = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(telemetry_mod)
InMemoryTelemetry = telemetry_mod.InMemoryTelemetry


def test_telemetry_records_and_snapshots():
    t = InMemoryTelemetry()
    t.record("/health", 200, 12.5)
    t.record("/predict", 500, 50.0)
    t.record("/health", 200, 10.0)

    snap = t.snapshot()

    assert snap["request_count"] == 3
    assert snap["status_counts"]["200"] == 2
    assert snap["status_counts"]["500"] == 1
    assert snap["path_counts"]["/health"] == 2
    assert snap["avg_latency_ms"] > 0
