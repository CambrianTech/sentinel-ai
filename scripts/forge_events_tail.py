#!/usr/bin/env python3
"""forge_events_tail.py — tail the forge event stream with formatted output.

Usage:
    python scripts/forge_events_tail.py [--root .factory] [--since TIMESTAMP]
                                        [--follow] [--format pretty|json]

This is the minimal v0 event subscriber that reads the `.events.jsonl`
sidecar from FACTORY-PROTOCOL.md v0.2 and emits human-readable lines.
Designed for operators and agents who want to watch a forge live
without sshing and tailing raw log files.

Replaces the polling-ssh pattern for live forge observation. Once
continuum's Events.emit() bridge is running, this script becomes a
reference implementation of how to consume the file-based stream;
the same output can come from subscribing to continuum's pub/sub.

Examples:

    # Follow live events from the default .factory/line location
    python scripts/forge_events_tail.py --follow

    # Show all events since a given timestamp
    python scripts/forge_events_tail.py --since 2026-04-10T00:00:00Z

    # Emit raw JSON for downstream tooling
    python scripts/forge_events_tail.py --follow --format json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def format_pretty(event: dict) -> str:
    """Format an event as a human-readable line."""
    ts = event.get("timestamp", "")
    host = event.get("host", "?")
    kind = event.get("kind", "?")
    alloy = event.get("alloy", "?")
    # Trim alloy name to last 50 chars for readability
    if len(alloy) > 50:
        alloy = "..." + alloy[-47:]

    # Short time (HH:MM:SS only)
    if "T" in ts and len(ts) >= 19:
        short_ts = ts[11:19]
    else:
        short_ts = ts[:8]

    # Kind-specific detail
    detail = ""
    if kind == "forge/started":
        stages = event.get("stages") or []
        source = event.get("source_model") or ""
        detail = f"  source={source}  stages={'→'.join(stages)}"
    elif kind == "forge/stage/started":
        stage = event.get("stage", "?")
        detail = f"  stage={stage}"
    elif kind == "forge/stage/progress":
        stage = event.get("stage", "?")
        substage = event.get("substage", "")
        progress = event.get("progress")
        samples_done = event.get("samples_done")
        samples_total = event.get("samples_total")
        if samples_done is not None and samples_total is not None:
            detail = f"  stage={stage}/{substage}  {samples_done}/{samples_total}"
        elif progress is not None:
            detail = f"  stage={stage}/{substage}  {progress*100:.1f}%"
        else:
            detail = f"  stage={stage}/{substage}"
    elif kind == "forge/stage/completed":
        stage = event.get("stage", "?")
        elapsed = event.get("elapsed_s", 0)
        detail = f"  stage={stage}  elapsed={elapsed}s"
    elif kind == "forge/model/load/started":
        src_gb = event.get("source_gb", 0)
        streaming = event.get("streaming", False)
        max_gpu = event.get("max_gpu_gb", 0)
        max_cpu = event.get("max_cpu_gb", 0)
        detail = f"  {src_gb:.1f}GB  streaming={streaming}  GPU≤{max_gpu}GiB  CPU≤{max_cpu}GiB"
    elif kind == "forge/model/load/completed":
        elapsed = event.get("elapsed_s", 0)
        peak_cpu = event.get("peak_cpu_gb", 0)
        peak_gpu = event.get("peak_gpu_gb", 0)
        detail = f"  elapsed={elapsed}s  peak_cpu={peak_cpu}GiB  peak_gpu={peak_gpu}GiB"
    elif kind == "forge/completed":
        elapsed = event.get("elapsed_s", 0)
        pmb_count = event.get("priorMetricBaselines_count", 0)
        published = event.get("published", False)
        hf_url = event.get("hf_repo_url", "")
        suffix = f"  hf={hf_url}" if hf_url else ""
        detail = f"  elapsed={elapsed}s  priorBaselines={pmb_count}  published={published}{suffix}"
    elif kind == "forge/rework":
        stage = event.get("stage", "?")
        error = event.get("error", "")[:80]
        detail = f"  stage={stage}  error={error}"

    return f"{short_ts}  {host:10s}  {kind:28s}  {alloy}{detail}"


def format_json(event: dict) -> str:
    return json.dumps(event, separators=(",", ":"))


def iter_events_from_file(path: Path, since: str | None = None):
    """Yield events from the file, skipping any older than `since`."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if since and event.get("timestamp", "") <= since:
            continue
        yield event


def follow_file(path: Path, since: str | None, formatter):
    """Tail -f the events file, emitting formatted events as they arrive.

    Not as efficient as tail -F since it re-reads the file on each poll,
    but simple and handles file rotation cleanly (rotate replaces the
    file, we re-read it). Good enough for a forge observer.
    """
    last_seen_ts = since or ""
    while True:
        try:
            for event in iter_events_from_file(path, since=last_seen_ts):
                print(formatter(event), flush=True)
                ts = event.get("timestamp", "")
                if ts > last_seen_ts:
                    last_seen_ts = ts
            time.sleep(1.0)
        except KeyboardInterrupt:
            break


def main():
    parser = argparse.ArgumentParser(description="Tail the forge event stream")
    parser.add_argument("--root", default=".factory", help="Factory root (default: .factory)")
    parser.add_argument("--line", default="line", help="Line name (default: line)")
    parser.add_argument("--since", default=None, help="Only show events newer than this ISO timestamp")
    parser.add_argument("--follow", "-f", action="store_true", help="Follow the file (tail -f)")
    parser.add_argument("--format", choices=["pretty", "json"], default="pretty", help="Output format")
    args = parser.parse_args()

    events_path = Path(args.root) / args.line / ".events.jsonl"
    formatter = format_pretty if args.format == "pretty" else format_json

    if args.follow:
        follow_file(events_path, args.since, formatter)
    else:
        for event in iter_events_from_file(events_path, since=args.since):
            print(formatter(event))


if __name__ == "__main__":
    main()
