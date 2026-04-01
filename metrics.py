"""
metrics.py — Append-only JSONL metrics storage.

Records per-reply metrics (latency, cost, tokens), pipeline results,
and user quality ratings. Provides simple aggregation for /metrics.
"""

import json
import time
from pathlib import Path


class MetricsStore:
    """Append-only JSONL metrics storage."""

    def __init__(self, path: Path):
        """Initialize with path to metrics.jsonl file."""
        self.path = Path(path)

    def record(self, event_type: str, data: dict) -> None:
        """Append a metrics event (one JSON line)."""
        entry = {
            "ts": time.time(),
            "type": event_type,
            **data,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def query(self, event_type: str | None = None, last_n: int = 100) -> list[dict]:
        """Read last N entries, optionally filtered by type."""
        if not self.path.exists():
            return []
        entries = []
        for line in self.path.read_text(encoding="utf-8").strip().splitlines():
            try:
                entry = json.loads(line)
                if event_type is None or entry.get("type") == event_type:
                    entries.append(entry)
            except json.JSONDecodeError:
                continue
        return entries[-last_n:]

    def summary(self) -> dict:
        """Compute aggregate stats: total cost, avg latency, rating, success rate."""
        entries = self.query()
        if not entries:
            return {"total_replies": 0, "message": "No data yet."}

        replies = [e for e in entries if e.get("type") == "reply"]
        pipelines = [e for e in entries if e.get("type") == "pipeline"]
        ratings = [e for e in entries if e.get("type") == "rating"]

        # Cost
        total_cost = 0.0
        for e in replies + pipelines:
            cost_str = e.get("cost", "")
            if isinstance(cost_str, str) and cost_str.startswith("$"):
                try:
                    total_cost += float(cost_str.lstrip("$"))
                except ValueError:
                    pass

        # Latency
        all_timed = [e for e in replies + pipelines if "time" in e or "latency" in e]
        avg_latency = 0.0
        if all_timed:
            latencies = [e.get("time", e.get("latency", 0)) for e in all_timed]
            avg_latency = sum(latencies) / len(latencies)

        # Quality (user ratings)
        avg_rating = None
        if ratings:
            scores = [e.get("score", 0) for e in ratings if isinstance(e.get("score"), (int, float))]
            if scores:
                avg_rating = round(sum(scores) / len(scores), 1)

        # Reliability (success rate for pipelines)
        success_rate = None
        if pipelines:
            ok = sum(1 for e in pipelines if e.get("status") == "ok")
            success_rate = f"{ok / len(pipelines) * 100:.0f}%"

        return {
            "total_replies": len(replies),
            "total_pipelines": len(pipelines),
            "total_cost": f"${total_cost:.4f}",
            "avg_latency": f"{avg_latency:.2f}s",
            "avg_rating": avg_rating,
            "total_ratings": len(ratings),
            "pipeline_success_rate": success_rate,
        }
