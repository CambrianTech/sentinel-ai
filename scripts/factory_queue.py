"""factory_queue — disk-backed queue + worker loop for the BigMama factory.

This is the loop that turns "we have alloys + a forge + an eval registry +
a publisher" into "BigMama cranks models 24/7." The simplest possible
disk-backed queue, no DB, no service, no network coordination.

Directory layout (created automatically under <root>/queue/):

    pending/    drop alloy files here (cp, mv, generator output)
    running/    worker moves alloy here while processing
    done/       success: alloy + .result.json sidecar
    failed/     failure: alloy + .error.json sidecar (error + traceback)

The filesystem IS the queue. Atomic moves give multi-worker safety for
free if we ever need to scale beyond a single 5090. The single-5090
case (today) uses one worker that processes one alloy at a time:
forge → eval → publish → mark done → next.

Public API:

    FactoryQueue(root)
        Wraps the directory layout. Methods:
            enqueue(alloy_path)               copy alloy into pending/
            pop_oldest_pending() → Path|None  atomic move pending → running
            mark_done(running_path, manifest) move running → done + manifest
            mark_failed(path, error, tb)      move running → failed + error
            stats() → dict                    counts of each bucket

    FactoryWorker(queue, executor, publisher, work_root)
        The loop that drains the queue. executor and publisher are injected
        callables (no GPU/HF dependencies in unit tests). Production wiring:
            executor  = alloy_executor.execute_alloy
            publisher = publish_model.publish

        Methods:
            process_one() → bool           pop, run, publish, mark; True if processed
            run_loop(max_iters=None) → int drain pending, return count processed

Production CLI (`python -m factory_queue --root .factory --max-iters 100`):
    Polls .factory/queue/pending/, processes each alloy end-to-end via the
    real executor + publisher, exits when pending is empty.

Reproducibility contract: this module is the dispatcher, NOT the source
of truth for forge behavior. The forge logic lives in alloy_executor.py
and the per-family adapters; this just routes alloys through them.
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Callable


class FactoryQueue:
    """Disk-backed queue at <root>/queue/{pending,running,done,failed}."""

    BUCKETS = ("pending", "running", "done", "failed")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.queue_dir = self.root / "queue"
        for bucket in self.BUCKETS:
            (self.queue_dir / bucket).mkdir(parents=True, exist_ok=True)

    @property
    def pending_dir(self) -> Path: return self.queue_dir / "pending"
    @property
    def running_dir(self) -> Path: return self.queue_dir / "running"
    @property
    def done_dir(self) -> Path:    return self.queue_dir / "done"
    @property
    def failed_dir(self) -> Path:  return self.queue_dir / "failed"

    def enqueue(self, alloy_path: str | Path) -> Path:
        """Copy an alloy file into pending/. Returns the path inside the queue."""
        src = Path(alloy_path)
        if not src.exists():
            raise FileNotFoundError(f"alloy file does not exist: {src}")
        dst = self.pending_dir / src.name
        shutil.copy2(src, dst)
        return dst

    def pop_oldest_pending(self) -> Path | None:
        """Atomically move the oldest pending alloy into running/.

        Returns the new path inside running/, or None if pending is empty.
        Atomic via rename — multi-worker safe (the worker that wins the
        rename is the one that processes the alloy; the loser sees
        FileNotFoundError on the next call and tries the next-oldest).
        """
        candidates = sorted(
            self.pending_dir.glob("*.alloy.json"),
            key=lambda p: p.stat().st_mtime,
        )
        for candidate in candidates:
            target = self.running_dir / candidate.name
            try:
                candidate.rename(target)
                return target
            except FileNotFoundError:
                # Another worker grabbed it; try the next one.
                continue
        return None

    def mark_done(self, running_path: Path, manifest: dict) -> Path:
        """Move running → done and write the result manifest sidecar."""
        if running_path.parent != self.running_dir:
            raise ValueError(
                f"mark_done called on path not in running/: {running_path}"
            )
        done_path = self.done_dir / running_path.name
        running_path.rename(done_path)
        sidecar = done_path.with_suffix(".result.json")
        sidecar.write_text(json.dumps({
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **manifest,
        }, indent=2))
        return done_path

    def mark_failed(
        self,
        running_path: Path,
        error: str,
        traceback_text: str = "",
    ) -> Path:
        """Move running → failed and write the error sidecar."""
        if running_path.parent != self.running_dir:
            raise ValueError(
                f"mark_failed called on path not in running/: {running_path}"
            )
        failed_path = self.failed_dir / running_path.name
        running_path.rename(failed_path)
        sidecar = failed_path.with_suffix(".error.json")
        sidecar.write_text(json.dumps({
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "error": error,
            "traceback": traceback_text,
        }, indent=2))
        return failed_path

    def stats(self) -> dict[str, int]:
        return {
            bucket: len(list((self.queue_dir / bucket).glob("*.alloy.json")))
            for bucket in self.BUCKETS
        }


class FactoryWorker:
    """The loop that drains a FactoryQueue.

    executor and publisher are injected callables so unit tests can pass
    fakes (no GPU/HF in unit tests). Production wiring is done by the
    CLI in main():
        executor  = alloy_executor.execute_alloy
        publisher = publish_model.publish
    """

    def __init__(
        self,
        queue: FactoryQueue,
        *,
        executor: Callable[..., Any],
        publisher: Callable[..., Any],
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
        """Pop the oldest pending alloy, forge → publish → mark.

        Returns True if an alloy was processed (success OR failure),
        False if pending was empty.
        """
        running = self.queue.pop_oldest_pending()
        if running is None:
            return False

        alloy_stem = running.stem.replace(".alloy", "")
        out_dir = self.work_root / alloy_stem

        # Stage 1: forge.
        try:
            executor_result = self.executor(str(running), output_dir=str(out_dir))
            # Some executors return the output dir, some don't; reconcile.
            forged_dir = Path(executor_result) if executor_result else out_dir
        except Exception as e:
            self.queue.mark_failed(
                running,
                error=f"executor failed: {e}",
                traceback_text=traceback.format_exc(),
            )
            return True

        # Stage 2: publish.
        try:
            publish_result = self.publisher(forged_dir, org=self.org)
        except Exception as e:
            self.queue.mark_failed(
                running,
                error=f"publisher failed: {e}",
                traceback_text=traceback.format_exc(),
            )
            return True

        # Stage 3: mark done with the manifest the publisher returned.
        manifest: dict = {"output_dir": str(forged_dir)}
        if isinstance(publish_result, str):
            manifest["hf_repo_url"] = publish_result
        elif isinstance(publish_result, dict):
            manifest.update(publish_result)
        self.queue.mark_done(running, manifest)
        return True

    def run_loop(
        self,
        max_iters: int | None = None,
        sleep_seconds: float = 0.0,
    ) -> int:
        """Drain pending until empty (or max_iters processed).

        Args:
            max_iters: process at most this many alloys before returning.
                       None = drain completely.
            sleep_seconds: how long to sleep between iterations when there's
                           still pending work. 0.0 for tests; production
                           should pass something small (~5s) so the worker
                           doesn't spin hot when pending is empty.

        Returns:
            Number of alloys processed (success + failure).
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
    """CLI entrypoint: drain a queue using the real executor + publisher.

    Usage:
        python -m factory_queue --root .factory --max-iters 10
    """
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".factory", help="queue root directory")
    ap.add_argument("--work-root", default=None, help="forge output root (default: <root>/work)")
    ap.add_argument("--org", default="continuum-ai")
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=5.0)
    args = ap.parse_args()

    from alloy_executor import execute_alloy
    from publish_model import publish

    queue = FactoryQueue(args.root)
    worker = FactoryWorker(
        queue,
        executor=execute_alloy,
        publisher=publish,
        work_root=args.work_root or (Path(args.root) / "work"),
        org=args.org,
    )
    print(f"factory worker starting at {args.root}, stats={queue.stats()}")
    n = worker.run_loop(max_iters=args.max_iters, sleep_seconds=args.sleep)
    print(f"processed {n} alloys, final stats={queue.stats()}")


if __name__ == "__main__":
    main()
