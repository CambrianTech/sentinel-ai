"""factory_queue — one hive node's local assembly line + daemon.

This module is ONE NODE of continuum's self-improving forge grid. The
grid (continuum) is the foreman: it decides which node builds which
alloy, when to pause, what fits in VRAM, what to prioritize. THIS code
is the dumb local executor that runs on each node:

  1. Watches intake/ for parts the grid placed there
  2. Builds them (forge → assay)
  3. Marks them finished/ or rework/ atomically
  4. Reports state via heartbeat the grid can poll
  5. Recovers stuck parts on startup so no work is ever lost
  6. Logs throughput so the grid sees the line history without scanning

Anything beyond this — foreman, priority, VRAM check, pause policy,
node selection, cancellation — lives in continuum's grid layer, NOT
here. The hive coordinator decides; the node just builds.

The daemon mode (`run_forever`) is the production shape: long-running,
poll-based, crash-safe via atomic disk moves. SIGTERM exits cleanly;
the next startup recovers any in-flight part via `recover_assembly()`.

Directory layout (created automatically under <root>/line/):

    intake/      drop alloy files here (cp, mv, generator output)
    assembly/    worker moves alloy here while building
    finished/    built + QA'd, sitting in the shipping bay for continuum
    rework/      flagged by QA, needs human attention (alloy + traceback)

CONTINUUM IS THE SHIPPING DEPARTMENT. Sentinel is the assembly + QA
floor. The shipping department reads finished/, applies its release
gates (alloy-declared minimum eval scores, security review, branding),
and ships to HuggingFace. Sentinel NEVER pushes to HF — that's a
deliberate architectural boundary. publisher injection exists only as
an opt-in for staging-environment integration tests; the production CLI
default is forge + eval, no push.

The filesystem IS the queue. Atomic moves give multi-worker safety for
free if you ever need to scale beyond a single 5090. The single-5090
case (today) uses one worker that processes one alloy at a time:
intake → assembly → finished → next.

Public API:

    FactoryQueue(root)
        Wraps the directory layout. Methods:
            enqueue(alloy_path)                  copy alloy into intake/
            pop_oldest_intake() → Path|None      atomic move intake → assembly
            mark_finished(assembly_path, ...)    move assembly → finished + manifest
            mark_rework(path, error, tb)         move assembly → rework + error
            stats() → dict                       counts of each station

    FactoryWorker(queue, executor, publisher, work_root)
        The line operator that drains the queue. executor is injected
        (the alloy runner). publisher is OPTIONAL and OFF by default —
        sentinel forges and assays, continuum ships. Production wiring:
            executor  = alloy_executor.execute_alloy
            publisher = None   # default; opt-in via --publish flag

        Methods:
            process_one() → bool           pop, build, mark; True if processed
            run_loop(max_iters=None) → int drain intake, return count processed

Production CLI (`python -m factory_queue --root .factory --max-iters 100`):
    Polls .factory/line/intake/, processes each alloy end-to-end via the
    real executor, marks finished, exits when intake is empty.

Reproducibility contract: this module is the line dispatcher, NOT the
source of truth for forge behavior. The forge logic lives in
alloy_executor.py and the per-family adapters; this just routes alloys
through them and records the assembly outcome.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Callable


# Maximum number of automatic retries for a part that gets stuck in
# assembly/. After this many retries, the part is moved to rework/
# permanently and a human (or continuum) must triage it.
MAX_RETRIES = 3


_RETRY_RE = re.compile(r"\.retry(\d+)\.alloy\.json$")


def _retry_count(filename: str) -> int:
    """How many times has this alloy been retried? Parsed from the
    filename marker .retry<N>.alloy.json. Fresh alloys have count 0."""
    m = _RETRY_RE.search(filename)
    return int(m.group(1)) if m else 0


def _bump_retry_filename(filename: str) -> str:
    """Increment the retry counter in an alloy filename, preserving stem."""
    n = _retry_count(filename)
    if n == 0:
        return filename.replace(".alloy.json", f".retry{n + 1}.alloy.json")
    return _RETRY_RE.sub(f".retry{n + 1}.alloy.json", filename)


def _is_pid_alive(pid: int) -> bool:
    """Cross-platform check whether a PID is alive."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


class FactoryQueue:
    """Disk-backed assembly line at <root>/line/{intake,assembly,finished,rework}.

    Stations:
        intake/    parts entering the line (alloys waiting to be built)
        assembly/  currently being built by the worker
        finished/  built + assayed, sitting in the shipping bay for continuum
        rework/    flagged by QA — alloy + traceback for human inspection
    """

    STATIONS = ("intake", "assembly", "finished", "rework")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.line_dir = self.root / "line"
        for station in self.STATIONS:
            (self.line_dir / station).mkdir(parents=True, exist_ok=True)

    @property
    def intake_dir(self) -> Path:    return self.line_dir / "intake"
    @property
    def assembly_dir(self) -> Path:  return self.line_dir / "assembly"
    @property
    def finished_dir(self) -> Path:  return self.line_dir / "finished"
    @property
    def rework_dir(self) -> Path:    return self.line_dir / "rework"

    # ── Daemon support: heartbeat, PID lock, throughput log ────────────
    @property
    def heartbeat_path(self) -> Path:        return self.line_dir / ".heartbeat.json"
    @property
    def pid_path(self) -> Path:              return self.line_dir / ".worker.pid"
    @property
    def throughput_log_path(self) -> Path:   return self.line_dir / "throughput.jsonl"

    def write_heartbeat(self, *, state: str, current_part: str | None) -> None:
        """Write the daemon heartbeat. The grid (continuum) polls this
        file to know what each node is doing without SSH."""
        data = {
            "pid": os.getpid(),
            "state": state,                            # idle | building | recovering | offline
            "current_part": current_part,
            "last_beat_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
        }
        # Atomic write: write to .tmp then rename, so a crashed write
        # never leaves a half-file the grid would mis-parse.
        tmp = self.heartbeat_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.rename(self.heartbeat_path)

    def read_heartbeat(self) -> dict | None:
        if not self.heartbeat_path.exists():
            return None
        try:
            return json.loads(self.heartbeat_path.read_text())
        except json.JSONDecodeError:
            return None

    def acquire_pid_lock(self) -> bool:
        """One worker per line. Returns True if we got the lock,
        False if a live worker is already running. Stale PID files
        (process dead) are cleaned automatically."""
        if self.pid_path.exists():
            try:
                existing_pid = int(self.pid_path.read_text().strip())
            except (ValueError, OSError):
                existing_pid = -1
            if _is_pid_alive(existing_pid):
                return False
            # Stale — clean it
            self.pid_path.unlink(missing_ok=True)
        self.pid_path.write_text(str(os.getpid()))
        return True

    def release_pid_lock(self) -> None:
        """Release the lock if we own it. Idempotent."""
        if not self.pid_path.exists():
            return
        try:
            owner = int(self.pid_path.read_text().strip())
        except (ValueError, OSError):
            owner = -1
        if owner == os.getpid():
            self.pid_path.unlink(missing_ok=True)

    def recover_assembly(self) -> list[Path]:
        """Recover any parts stuck in assembly/ from a previous worker.

        For each stuck part:
          - If retry count < MAX_RETRIES: bump the counter, move back to
            intake/ for another attempt
          - If retry count >= MAX_RETRIES: move to rework/ permanently
            with an error sidecar explaining why

        Returns the list of recovered (now-relocated) part paths.
        Called on daemon startup so no work is ever lost to a crash.
        """
        recovered: list[Path] = []
        for stuck in sorted(self.assembly_dir.glob("*.alloy.json")):
            retries = _retry_count(stuck.name)
            if retries >= MAX_RETRIES:
                target = self.rework_dir / stuck.name
                stuck.rename(target)
                sidecar = target.with_suffix(".error.json")
                sidecar.write_text(json.dumps({
                    "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "error": f"crash recovery: exceeded MAX_RETRIES={MAX_RETRIES}",
                    "traceback": "",
                    "retries": retries,
                }, indent=2))
                self._append_throughput({
                    "outcome": "rework",
                    "alloy": stuck.name,
                    "error": "exceeded MAX_RETRIES on crash recovery",
                    "retries": retries,
                })
                recovered.append(target)
            else:
                new_name = _bump_retry_filename(stuck.name)
                target = self.intake_dir / new_name
                stuck.rename(target)
                self._append_throughput({
                    "outcome": "recovered",
                    "alloy": new_name,
                    "retries": retries + 1,
                })
                recovered.append(target)
        return recovered

    def status(self) -> dict:
        """Read heartbeat + queue stats and return a summary dict.

        Used by the CLI `--status` command and (eventually) by
        continuum's grid view of the line. Doesn't require the worker
        to be running — when there's no heartbeat OR the heartbeat's
        PID is dead, state is 'offline' and the stats are still
        accurate. The dead-pid case happens after a hard crash: the
        heartbeat file is sticky on disk and would otherwise lie about
        the daemon being alive.
        """
        hb = self.read_heartbeat()
        if hb is None:
            return {
                "state": "offline",
                "stats": self.stats(),
                "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
            }
        # Stale-heartbeat detection: if the recorded pid isn't alive,
        # the daemon crashed and left the file behind. Don't lie.
        recorded_pid = hb.get("pid")
        if isinstance(recorded_pid, int) and not _is_pid_alive(recorded_pid):
            return {
                **hb,
                "state": "offline",
                "stale_heartbeat": True,
                "stats": self.stats(),
            }
        return {
            **hb,
            "stats": self.stats(),
        }

    def _append_throughput(self, entry: dict) -> None:
        """Append-only JSONL log of every state transition. Continuum
        reads this for the shipping dashboard / grid view."""
        entry = {
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **entry,
        }
        with open(self.throughput_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def enqueue(self, alloy_path: str | Path) -> Path:
        """Copy an alloy file into intake/. Returns the path inside the line."""
        src = Path(alloy_path)
        if not src.exists():
            raise FileNotFoundError(f"alloy file does not exist: {src}")
        dst = self.intake_dir / src.name
        shutil.copy2(src, dst)
        return dst

    def pop_oldest_intake(self) -> Path | None:
        """Atomically move the oldest intake alloy onto the assembly line.

        Returns the new path inside assembly/, or None if intake is empty.
        Atomic via rename — multi-worker safe (the worker that wins the
        rename is the one that builds it; losers see FileNotFoundError on
        the next call and try the next-oldest part).
        """
        candidates = sorted(
            self.intake_dir.glob("*.alloy.json"),
            key=lambda p: p.stat().st_mtime,
        )
        for candidate in candidates:
            target = self.assembly_dir / candidate.name
            try:
                candidate.rename(target)
                return target
            except FileNotFoundError:
                # Another worker grabbed it; try the next one.
                continue
        return None

    def mark_finished(self, assembly_path: Path, manifest: dict) -> Path:
        """Move assembly → finished and write the result manifest sidecar.

        The manifest is what the shipping department (continuum) reads to
        decide ship/rework. It points at the on-disk forged artifact and
        carries the eval results so the release gates can fire.
        """
        if assembly_path.parent != self.assembly_dir:
            raise ValueError(
                f"mark_finished called on path not in assembly/: {assembly_path}"
            )
        finished_path = self.finished_dir / assembly_path.name
        assembly_path.rename(finished_path)
        sidecar = finished_path.with_suffix(".result.json")
        sidecar.write_text(json.dumps({
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **manifest,
        }, indent=2))
        self._append_throughput({
            "outcome": "finished",
            "alloy": finished_path.name,
            **{k: v for k, v in manifest.items() if k in ("forged_dir", "hf_repo_url")},
        })
        return finished_path

    def mark_rework(
        self,
        assembly_path: Path,
        error: str,
        traceback_text: str = "",
    ) -> Path:
        """Move assembly → rework and write the error sidecar.

        Rework is for parts that failed QA: a human inspects the traceback
        and either fixes the recipe + re-queues at intake/, or scraps it.
        """
        if assembly_path.parent != self.assembly_dir:
            raise ValueError(
                f"mark_rework called on path not in assembly/: {assembly_path}"
            )
        rework_path = self.rework_dir / assembly_path.name
        assembly_path.rename(rework_path)
        sidecar = rework_path.with_suffix(".error.json")
        sidecar.write_text(json.dumps({
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "error": error,
            "traceback": traceback_text,
        }, indent=2))
        self._append_throughput({
            "outcome": "rework",
            "alloy": rework_path.name,
            "error": error,
        })
        return rework_path

    def stats(self) -> dict[str, int]:
        return {
            station: len(list((self.line_dir / station).glob("*.alloy.json")))
            for station in self.STATIONS
        }


class FactoryWorker:
    """The loop that drains a FactoryQueue.

    Sentinel-ai's job is FORGE + EVAL. Publication is continuum's
    responsibility — the worker writes the forged artifact + score sheet
    to done/ and stops there. Continuum picks up done/ items, applies
    quality gates, reviews, and (separately) publishes. Sentinel never
    pushes to HuggingFace; that's a deliberate architectural boundary.

    executor is the only required callable. publisher exists as an
    OPTIONAL injected callable for the rare case where a queue is
    explicitly run with publish enabled (e.g. an integration test that
    pushes to a private staging repo). Default is None — forge + eval,
    no push.
    """

    def __init__(
        self,
        queue: FactoryQueue,
        *,
        executor: Callable[..., Any],
        publisher: Callable[..., Any] | None = None,
        work_root: str | Path,
        org: str = "continuum-ai",
    ) -> None:
        self.queue = queue
        self.executor = executor
        self.publisher = publisher
        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.org = org

    def process_one(self) -> bool:
        """Pop the oldest intake alloy, forge → assay → mark finished.

        On success, the forged artifact dir is written to finished/ with
        a sidecar manifest pointing at the on-disk artifact and the eval
        results. Continuum (the shipping department) reads finished/ and
        applies its release gates.

        Returns True if a part was processed (success OR rework), False
        if intake was empty.
        """
        assembly = self.queue.pop_oldest_intake()
        if assembly is None:
            return False

        alloy_stem = assembly.stem.replace(".alloy", "")
        forged_root = self.work_root / alloy_stem
        self.queue.write_heartbeat(state="building", current_part=assembly.name)

        # Station 1: forge + assay. The alloy's eval stage runs through
        # the registered BenchmarkRunner pack as part of execute_alloy.
        try:
            executor_result = self.executor(str(assembly), output_dir=str(forged_root))
            forged_dir = Path(executor_result) if executor_result else forged_root
        except Exception as e:
            self.queue.mark_rework(
                assembly,
                error=f"executor failed: {e}",
                traceback_text=traceback.format_exc(),
            )
            return True

        # Station 2 (OPTIONAL): publish. Default is OFF. Continuum is
        # the shipping department. The worker only invokes the publisher
        # if one was explicitly injected (staging-environment test path).
        publish_manifest: dict = {}
        if self.publisher is not None:
            try:
                publish_result = self.publisher(forged_dir, org=self.org)
            except Exception as e:
                self.queue.mark_rework(
                    assembly,
                    error=f"publisher failed: {e}",
                    traceback_text=traceback.format_exc(),
                )
                return True
            if isinstance(publish_result, str):
                publish_manifest["hf_repo_url"] = publish_result
            elif isinstance(publish_result, dict):
                publish_manifest.update(publish_result)

        # Station 3: mark finished. The manifest tells continuum where the
        # forged artifact lives on disk so the shipping flow there can
        # read the alloy + eval results and apply its release gates.
        manifest: dict = {
            "forged_dir": str(forged_dir),
            "alloy_path": str(assembly),
            "published": self.publisher is not None,
            **publish_manifest,
        }
        self.queue.mark_finished(assembly, manifest)
        return True

    def run_forever(
        self,
        idle_sleep_seconds: float = 5.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> int:
        """Daemon mode — poll intake/ forever, never exit on empty.

        This is the production shape: long-running, crash-safe, clean
        SIGTERM handling. The hive node stays online ready to receive
        work the grid pushes into intake/. When intake is empty the
        worker idles for `idle_sleep_seconds` and tries again.

        On startup it ALWAYS calls recover_assembly() first so any part
        a previous worker died on gets re-queued (or moved permanently
        to rework/ if it has exhausted its retries).

        sleep_fn is injectable for tests — production passes time.sleep,
        tests pass a function that records calls and raises after N
        idle iterations to terminate cleanly.

        Returns the count of parts processed before the loop exited
        (KeyboardInterrupt or SIGTERM).
        """
        self.queue.recover_assembly()
        self.queue.write_heartbeat(state="idle", current_part=None)
        processed = 0
        try:
            while True:
                did_work = self.process_one()
                if did_work:
                    processed += 1
                    self.queue.write_heartbeat(state="idle", current_part=None)
                    continue
                # Intake empty — idle and try again
                self.queue.write_heartbeat(state="idle", current_part=None)
                sleep_fn(idle_sleep_seconds)
        except (KeyboardInterrupt, SystemExit):
            self.queue.write_heartbeat(state="offline", current_part=None)
        return processed

    def run_loop(
        self,
        max_iters: int | None = None,
        sleep_seconds: float = 0.0,
    ) -> int:
        """Drain intake until empty (or max_iters processed).

        Args:
            max_iters: process at most this many parts before returning.
                       None = drain completely.
            sleep_seconds: how long to sleep between iterations when intake
                           is non-empty. 0.0 for tests; production should
                           pass something small (~5s) so the worker doesn't
                           spin hot when intake is empty.

        Returns:
            Number of parts processed (finished + rework).
        """
        processed = 0
        while True:
            if max_iters is not None and processed >= max_iters:
                break
            did_work = self.process_one()
            if not did_work:
                break
            processed += 1
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        return processed


def main():
    """CLI entrypoint — manual controls for one hive node.

    Subcommands:
        (default)        Run as a daemon: poll intake/ forever, recover
                         crashed assembly/ parts on startup, build each
                         part through alloy_executor, mark finished/rework.
                         SIGTERM/Ctrl-C exits cleanly.

        --status         Print current line state + heartbeat (no daemon
                         needed; reads .heartbeat.json + bucket counts).
        --recover        One-shot crash recovery: scan assembly/, push
                         stuck parts back to intake/ (or rework/ if
                         retries exhausted). Useful before starting a
                         daemon you suspect crashed.
        --tail           Print the last 20 throughput.jsonl entries.
        --max-iters N    Process at most N parts then exit (testing).
        --once           Process exactly one part then exit (testing).

    The grid (continuum) is the foreman. This CLI is for direct local
    operation when you need to poke a node without going through the
    grid command surface.
    """
    import argparse
    import signal

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".factory", help="queue root directory")
    ap.add_argument("--work-root", default=None, help="forge output root (default: <root>/work)")
    ap.add_argument("--org", default="continuum-ai")
    ap.add_argument("--idle-sleep", type=float, default=5.0,
                    help="seconds to sleep between intake polls when empty")
    ap.add_argument("--max-iters", type=int, default=None,
                    help="testing: process at most N parts then exit")
    ap.add_argument("--once", action="store_true",
                    help="testing: process exactly one part then exit")
    ap.add_argument("--status", action="store_true",
                    help="print current state + heartbeat and exit")
    ap.add_argument("--recover", action="store_true",
                    help="one-shot crash recovery: drain assembly/ and exit")
    ap.add_argument("--tail", action="store_true",
                    help="print the last 20 throughput.jsonl entries and exit")
    ap.add_argument(
        "--publish",
        action="store_true",
        help=(
            "Opt-in: also push to HuggingFace after forge+eval. Default OFF — "
            "sentinel forges and evals; continuum is the publication gatekeeper."
        ),
    )
    args = ap.parse_args()

    queue = FactoryQueue(args.root)

    # ── Read-only commands (no daemon) ──────────────────────────────────────
    if args.status:
        s = queue.status()
        print(json.dumps(s, indent=2))
        return

    if args.tail:
        if not queue.throughput_log_path.exists():
            print("(no throughput log yet)")
            return
        lines = queue.throughput_log_path.read_text().splitlines()[-20:]
        for line in lines:
            print(line)
        return

    if args.recover:
        recovered = queue.recover_assembly()
        print(f"recovered {len(recovered)} stuck parts")
        for p in recovered:
            print(f"  {p.relative_to(queue.line_dir)}")
        return

    # ── Daemon mode ─────────────────────────────────────────────────────────
    from alloy_executor import execute_alloy

    publisher = None
    if args.publish:
        from publish_model import publish as publisher

    if not queue.acquire_pid_lock():
        existing = queue.pid_path.read_text().strip()
        print(f"another worker is already running (pid {existing}). exiting.")
        return

    # SIGTERM → KeyboardInterrupt so run_forever exits cleanly via the
    # same path as Ctrl-C. The grid (continuum) sends SIGTERM when it
    # wants to gracefully shut down a node.
    def _sigterm(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _sigterm)

    worker = FactoryWorker(
        queue,
        executor=execute_alloy,
        publisher=publisher,
        work_root=args.work_root or (Path(args.root) / "work"),
        org=args.org,
    )

    print(f"hive node starting at {args.root}")
    print(f"  host: {os.uname().nodename if hasattr(os, 'uname') else 'unknown'}")
    print(f"  pid:  {os.getpid()}")
    print(f"  stats: {queue.stats()}")
    print(f"  publish: {'ON' if publisher is not None else 'OFF (continuum is the shipping department)'}")
    try:
        if args.once:
            queue.recover_assembly()
            worker.process_one()
            queue.write_heartbeat(state="offline", current_part=None)
        elif args.max_iters is not None:
            queue.recover_assembly()
            n = worker.run_loop(max_iters=args.max_iters, sleep_seconds=args.idle_sleep)
            queue.write_heartbeat(state="offline", current_part=None)
            print(f"processed {n} parts, final stats={queue.stats()}")
        else:
            n = worker.run_forever(idle_sleep_seconds=args.idle_sleep)
            print(f"processed {n} parts before shutdown, final stats={queue.stats()}")
    finally:
        queue.release_pid_lock()


if __name__ == "__main__":
    main()
