"""factory_queue — disk-backed assembly line for the BigMama factory.

This is the production line that turns "we have alloys + a forge + an
eval registry" into "BigMama cranks models 24/7." Toyota Production
System over alchemy: parts enter intake, move down the assembly line,
get QA'd, and end up in the shipping bay (or rework).

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
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Callable


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
    ap.add_argument(
        "--publish",
        action="store_true",
        help=(
            "Opt-in: also push to HuggingFace after forge+eval. Default OFF — "
            "sentinel forges and evals; continuum is the publication gatekeeper."
        ),
    )
    args = ap.parse_args()

    from alloy_executor import execute_alloy

    publisher = None
    if args.publish:
        # Opt-in only. Default is forge + eval, no HF push. Continuum
        # is the publication gatekeeper for production runs.
        from publish_model import publish as publisher

    queue = FactoryQueue(args.root)
    worker = FactoryWorker(
        queue,
        executor=execute_alloy,
        publisher=publisher,
        work_root=args.work_root or (Path(args.root) / "work"),
        org=args.org,
    )
    print(f"factory worker starting at {args.root}, stats={queue.stats()}")
    n = worker.run_loop(max_iters=args.max_iters, sleep_seconds=args.sleep)
    print(f"processed {n} alloys, final stats={queue.stats()}")


if __name__ == "__main__":
    main()
