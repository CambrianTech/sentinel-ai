"""Output stages — end of the pipeline.

These produce deliverables:
- Quant: GGUF/MLX/ONNX quantization
- Package: device-specific packaging (CoreML, TensorRT)
- Eval: benchmark evaluation (HumanEval, MMLU, etc.)
- Publish: push to HuggingFace
- Deploy: push to grid node
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from .base import StageExecutor, ForgeContext


class QuantExecutor(StageExecutor):
    """Quantize model to GGUF/MLX/ONNX/safetensors."""

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        fmt = self.config.get("format", "gguf")
        quant_types = self.config.get("quantTypes", ["Q4_K_M"])
        device_targets = self.config.get("deviceTargets", [])

        self.log(f"Quantizing to {fmt}: {', '.join(quant_types)}")
        if device_targets:
            self.log(f"  Device targets: {', '.join(device_targets)}")

        model_dir = ctx.output_dir / "model"
        if not model_dir.exists():
            self.log("WARNING: No model directory — quant deferred")
            return ctx

        if fmt == "gguf":
            for qt in quant_types:
                out_path = ctx.output_dir / f"{ctx.alloy.get('name', 'model')}-{qt}.gguf"
                self.log(f"  {qt} → {out_path.name}")
            self.log("  Requires: llama.cpp (convert_hf_to_gguf.py + llama-quantize)")
            self.log(f"  Use: python scripts/publish_gguf.py {model_dir}")
        elif fmt == "mlx":
            self.log("  Requires: mlx-lm (pip install mlx-lm)")
            self.log(f"  Use: mlx_lm.convert --hf-path {model_dir} --q-bits 4")
        elif fmt == "onnx":
            self.log("  Requires: optimum (pip install optimum)")
            self.log(f"  Use: optimum-cli export onnx -m {model_dir}")
        else:
            self.log(f"  {fmt} not yet supported")

        return ctx


class PackageExecutor(StageExecutor):
    """Package for device-specific runtimes beyond basic quantization."""

    PACKAGE_TOOLS = {
        "coreml": "coremltools (pip install coremltools)",
        "tensorrt": "tensorrt-llm (pip install tensorrt-llm)",
        "onnx": "optimum (pip install optimum[onnxruntime])",
        "openvino": "optimum-intel (pip install optimum[openvino])",
    }

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        fmt = self.config.get("format", "coreml")
        runtime = self.config.get("runtime", "")
        optimization = self.config.get("optimization", "balanced")

        self.log(f"Packaging for {fmt}" + (f" ({runtime})" if runtime else ""))
        self.log(f"  Optimization: {optimization}")

        tool = self.PACKAGE_TOOLS.get(fmt, f"Unknown tool for {fmt}")
        self.log(f"  Requires: {tool}")

        return ctx


class EvalExecutor(StageExecutor):
    """Run benchmark evaluations.

    Supports:
    - humaneval/humaneval+: via evalplus (pip install evalplus)
    - mmlu, gsm8k, arc, hellaswag, winogrande, truthfulqa: via lm-eval-harness
    """

    # Benchmark → harness mapping
    EVALPLUS_BENCHMARKS = {"humaneval", "humaneval+"}
    LM_EVAL_BENCHMARKS = {"mmlu", "gsm8k", "arc", "arc_challenge", "hellaswag",
                          "winogrande", "truthfulqa"}

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        benchmarks = self.config.get("benchmarks", [])
        compare = self.config.get("compareToBase", True)

        if not benchmarks:
            self.log("No benchmarks specified — skipping")
            return ctx

        model_dir = ctx.output_dir / "model"
        if not model_dir.exists():
            self.log("WARNING: No model directory — eval deferred")
            for bench in benchmarks:
                ctx.eval_results.append({
                    "name": bench["name"],
                    "metrics": {"status": "deferred", "reason": "no model directory"},
                })
            return ctx

        for bench in benchmarks:
            name = bench["name"]
            self.log(f"Running {name}...")

            if name in self.EVALPLUS_BENCHMARKS:
                result = self._run_evalplus(ctx, name, model_dir)
            elif name in self.LM_EVAL_BENCHMARKS:
                result = self._run_lm_eval(ctx, name, model_dir)
            else:
                self.log(f"  Unknown benchmark: {name} — recording as pending")
                result = {"name": name, "metrics": {"status": "pending"}}

            ctx.eval_results.append(result)

        return ctx

    def _run_evalplus(self, ctx: ForgeContext, name: str, model_dir: Path) -> dict:
        """Run HumanEval/HumanEval+ via evalplus."""
        try:
            import evalplus
        except ImportError:
            raise RuntimeError("evalplus not installed. Run: .venv/bin/pip install evalplus")

        result_dir = ctx.output_dir / "eval" / name
        result_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Generate completions — GPU batch inference, greedy decoding
            self.log(f"  Generating completions for {name} (GPU batch)...")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
            subprocess.check_call([
                sys.executable, "-m", "evalplus.codegen",
                "--model", str(model_dir),
                "--dataset", "humaneval",
                "--backend", "hf",
                "--greedy",
                "--bs", "8",
                "--output-path", str(result_dir),
            ], timeout=3600, env=env)  # 1 hour timeout (GPU is fast)

            # Evaluate
            self.log(f"  Evaluating completions...")
            eval_result = subprocess.run([
                sys.executable, "-m", "evalplus.evaluate",
                "--dataset", "humaneval",
                "--samples", str(result_dir),
            ], capture_output=True, text=True, timeout=3600)

            # Parse evalplus output
            metrics = self._parse_evalplus_output(eval_result.stdout, name)
            result_hash = self._hash_directory(result_dir)

            self.log(f"  {name}: {metrics.get('passing', '?')}/{metrics.get('total', '?')} "
                     f"({metrics.get('score', '?')}%)")

            return {
                "name": name,
                "metrics": metrics,
                "resultHash": result_hash,
                "submittedToLeaderboard": False,
            }

        except subprocess.TimeoutExpired:
            self.log(f"  {name} timed out")
            return {"name": name, "metrics": {"status": "timeout"}}
        except Exception as e:
            self.log(f"  {name} failed: {e}")
            return {"name": name, "metrics": {"status": "failed", "error": str(e)}}

    def _run_lm_eval(self, ctx: ForgeContext, name: str, model_dir: Path) -> dict:
        """Run benchmark via lm-eval-harness."""
        try:
            import lm_eval
        except ImportError:
            raise RuntimeError("lm-eval not installed. Run: .venv/bin/pip install lm-eval")

        result_dir = ctx.output_dir / "eval" / name
        result_dir.mkdir(parents=True, exist_ok=True)

        # Map our names to lm-eval task names
        task_map = {
            "mmlu": "mmlu",
            "gsm8k": "gsm8k",
            "arc": "arc_challenge",
            "arc_challenge": "arc_challenge",
            "hellaswag": "hellaswag",
            "winogrande": "winogrande",
            "truthfulqa": "truthfulqa_mc2",
        }
        task = task_map.get(name, name)

        try:
            self.log(f"  Running lm-eval task: {task}")
            eval_result = subprocess.run([
                sys.executable, "-m", "lm_eval",
                "--model", "hf",
                "--model_args", f"pretrained={model_dir}",
                "--tasks", task,
                "--batch_size", "auto",
                "--output_path", str(result_dir),
            ], capture_output=True, text=True, timeout=7200)

            # Parse lm-eval results
            metrics = self._parse_lm_eval_output(result_dir, task)
            result_hash = self._hash_directory(result_dir)

            self.log(f"  {name}: {metrics}")

            return {
                "name": name,
                "metrics": metrics,
                "resultHash": result_hash,
                "submittedToLeaderboard": False,
            }

        except subprocess.TimeoutExpired:
            self.log(f"  {name} timed out")
            return {"name": name, "metrics": {"status": "timeout"}}
        except Exception as e:
            self.log(f"  {name} failed: {e}")
            return {"name": name, "metrics": {"status": "failed", "error": str(e)}}

    def _parse_evalplus_output(self, output: str, name: str) -> dict:
        """Parse evalplus stdout for the pass@1 of a SPECIFIC benchmark.

        evalplus's CLI prints both base and plus scores in one run:
            humaneval (base tests)
            pass@1:	0.884
            humaneval+ (base + extra tests)
            pass@1:	0.854

        This parser must select the RIGHT pass@1 line for the benchmark
        we're scoring. Previous version walked all lines and overwrote
        metrics["score"] each iteration, so it always returned the LAST
        pass@1 (humaneval_plus) regardless of which benchmark `name` was.
        That assigned the humaneval_plus value to a humaneval benchmark.
        Fix: section-aware parsing.

        canonical pass@1 from evalplus:
            humaneval (base tests):       (tasks where base_status == 'pass') / total
            humaneval+ (base+extra tests): (tasks where base_status == plus_status == 'pass') / total
        Both are computed by evalplus.estimate_pass_at_k internally; we
        just read the printed values.
        """
        import re
        metrics: dict = {}

        if name in ("humaneval", "humaneval_plus", "humaneval+"):
            # Section header: "humaneval (base tests)" then "pass@1:\t<float>"
            # Then:           "humaneval+ (base + extra tests)" then next "pass@1:\t<float>"
            base_match = re.search(
                r"humaneval \(base tests\)\s*\n?\s*pass@1:\s*(\d+\.\d+)",
                output,
            )
            plus_match = re.search(
                r"humaneval\+ \(base \+ extra tests\)\s*\n?\s*pass@1:\s*(\d+\.\d+)",
                output,
            )
            if name == "humaneval" and base_match:
                pass1 = float(base_match.group(1))
                metrics["score"] = round(pass1 * 100, 2)
                metrics["pass_at_1_fraction"] = pass1
            elif name in ("humaneval_plus", "humaneval+") and plus_match:
                pass1 = float(plus_match.group(1))
                metrics["score"] = round(pass1 * 100, 2)
                metrics["pass_at_1_fraction"] = pass1
            else:
                metrics["status"] = "completed_no_parse"
                metrics["raw_output"] = output[-500:]
            return metrics

        # Fallback: not a HumanEval-family benchmark, walk lines naively.
        # Some lm-eval-harness benchmarks print pass@1 in different formats.
        # If we see one and it's parseable, use it; otherwise record raw.
        for line in output.splitlines():
            line = line.strip()
            if "pass@1" in line.lower():
                parts = line.split(":")
                if len(parts) >= 2:
                    val = parts[-1].strip()
                    if "/" in val:
                        passing, total = val.split("/")
                        metrics["passing"] = int(passing.strip())
                        metrics["total"] = int(total.strip())
                        metrics["score"] = round(metrics["passing"] / metrics["total"] * 100, 2)
                    else:
                        try:
                            score = float(val)
                            metrics["score"] = round(score * 100, 2) if score <= 1.0 else round(score, 2)
                        except ValueError:
                            pass
        if not metrics:
            metrics["status"] = "completed_no_parse"
            metrics["raw_output"] = output[-500:]
        return metrics

    def _parse_lm_eval_output(self, result_dir: Path, task: str) -> dict:
        """Parse lm-eval JSON results."""
        # lm-eval writes results to a JSON file in the output directory
        for json_file in result_dir.rglob("*.json"):
            try:
                data = json.loads(json_file.read_text())
                results = data.get("results", {})
                if task in results:
                    task_results = results[task]
                    # Extract the primary metric
                    acc = task_results.get("acc,none") or task_results.get("acc_norm,none")
                    if acc is not None:
                        return {
                            "accuracy": round(acc * 100, 1),
                            "nShot": data.get("config", {}).get("num_fewshot", 0),
                        }
                    # Return whatever metrics exist
                    return {k: v for k, v in task_results.items()
                            if not k.startswith("_") and v is not None}
            except (json.JSONDecodeError, KeyError):
                continue
        return {"status": "completed_no_results"}

    def _hash_directory(self, directory: Path) -> str:
        """SHA-256 hash of all files in a directory (deterministic)."""
        h = hashlib.sha256()
        for f in sorted(directory.rglob("*")):
            if f.is_file():
                h.update(f.read_bytes())
        return f"sha256:{h.hexdigest()}"


class DeliverExecutor(StageExecutor):
    """Deliver forge results — write delivery manifest, do NOT auto-publish.

    Writes delivery.json with results summary so the owner can review
    before publishing. Use publish_model.py to actually upload to HF.
    """

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        # ctx.alloy.get("results") may legitimately be None for fresh
        # forges that haven't been re-saved through the publish stage
        # yet. Treat None the same as missing — empty results dict.
        r = ctx.alloy.get("results") or {}
        benchmarks = r.get("benchmarks", [])

        # Compute model size
        model_dir = ctx.output_dir / "model"
        model_size_gb = 0
        if model_dir.exists():
            model_size_gb = round(sum(
                f.stat().st_size for f in model_dir.glob("*.safetensors")
            ) / (1024 ** 3), 2)

        delivery = {
            "completedAt": datetime.now(timezone.utc).isoformat(),
            "alloyName": ctx.alloy.get("name", "unknown"),
            "outputDir": str(ctx.output_dir),
            "summary": {
                "baselinePerplexity": r.get("baselinePerplexity"),
                "finalPerplexity": r.get("finalPerplexity"),
                "improvementPct": r.get("improvementPct"),
                "benchmarks": [
                    {"name": b.get("name"), "score": b.get("metrics", {}).get("score")}
                    for b in benchmarks
                ],
                "modelSizeGb": model_size_gb,
            },
            "publishReady": True,
        }

        delivery_path = ctx.output_dir / "delivery.json"
        delivery_path.write_text(json.dumps(delivery, indent=2))
        self.log(f"Delivery manifest: {delivery_path}")
        self.log(f"  Perplexity: {r.get('baselinePerplexity')} → {r.get('finalPerplexity')}")
        self.log(f"  Model size: {model_size_gb}GB")
        self.log(f"")
        self.log(f"  To publish: python scripts/publish_model.py {ctx.output_dir}")
        return ctx


class PublishExecutor(StageExecutor):
    """DEPRECATED: Use DeliverExecutor + publish_model.py instead.

    Kept for backward compatibility. Now delegates to DeliverExecutor
    and logs a deprecation warning.
    """

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        self.log("WARNING: 'publish' stage is deprecated. Use 'deliver' + publish_model.py")
        self.log("  Writing delivery manifest (publish_model.py does the actual upload)")

        # Write delivery manifest first
        deliver = DeliverExecutor(self.config)
        ctx = deliver.execute(ctx)

        # Then do the actual publish for backward compat
        org = self.config.get("org", "continuum-ai")
        include_alloy = self.config.get("includeAlloy", True)
        tags = self.config.get("tags", [])
        private = self.config.get("private", False)

        model_dir = ctx.output_dir / "model"
        if not model_dir.exists():
            self.log("WARNING: No model directory — publish deferred")
            return ctx

        # Build repo name from alloy
        alloy = ctx.alloy
        name = alloy.get("name", ctx.model_name.split("/")[-1])
        repo_id = f"{org}/{name}"
        pub_url = f"https://huggingface.co/{repo_id}"
        pub_time = datetime.now(timezone.utc).isoformat()

        self.log(f"Publishing to {repo_id}")

        # Verify alloy integrity before publishing
        errors = self._verify_integrity(ctx)
        if errors:
            for err in errors:
                self.log(f"  INTEGRITY ERROR: {err}")
            self.log("  ABORTING PUBLISH — integrity verification failed")
            return ctx

        try:
            from huggingface_hub import HfApi, create_repo
        except ImportError:
            self.log("  huggingface_hub not installed — install with: pip install huggingface_hub")
            self.log(f"  Manual: huggingface-cli upload {repo_id} {ctx.output_dir}")
            return ctx

        api = HfApi()

        # Create repo
        try:
            create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
            self.log(f"  Repo ready: {repo_id}")
        except Exception as e:
            self.log(f"  ERROR creating repo: {e}")
            return ctx

        # --- PHASE 1: Upload model weights (these don't affect hashes) ---
        safetensors = list(model_dir.glob("*.safetensors"))
        if safetensors:
            self.log(f"  Uploading {len(safetensors)} weight files...")
            for sf in safetensors:
                api.upload_file(path_or_fileobj=str(sf), path_in_repo=sf.name, repo_id=repo_id)

            for cfg in ["config.json", "tokenizer.json", "tokenizer_config.json",
                        "generation_config.json", "special_tokens_map.json"]:
                cfg_path = model_dir / cfg
                if cfg_path.exists():
                    api.upload_file(path_or_fileobj=str(cfg_path), path_in_repo=cfg, repo_id=repo_id)

        # Upload benchmark samples
        bench_dir = ctx.output_dir / "benchmark"
        if bench_dir.exists():
            for txt in bench_dir.glob("*.txt"):
                api.upload_file(path_or_fileobj=str(txt),
                                path_in_repo=f"benchmark/{txt.name}", repo_id=repo_id)

        # Upload eval results
        eval_dir = ctx.output_dir / "eval"
        if eval_dir.exists():
            for f in eval_dir.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(ctx.output_dir)
                    api.upload_file(path_or_fileobj=str(f), path_in_repo=str(rel), repo_id=repo_id)

        # --- PHASE 2: Finalize alloy (write receipt FIRST, then hash) ---
        alloy_files = list(ctx.output_dir.glob("*.alloy.json"))
        if alloy_files and include_alloy:
            alloy_data = json.loads(alloy_files[0].read_text())
            alloy_data["receipt"] = {
                "publications": [{
                    "target": "huggingface",
                    "url": pub_url,
                    "publishedAt": pub_time,
                }],
                "issuedAt": pub_time,
            }
            # Save finalized alloy — this is the version that gets hashed
            alloy_files[0].write_text(json.dumps(alloy_data, indent=2))
            ctx.alloy = alloy_data  # Update context for card generation

        # --- PHASE 3: Hash the FINAL alloy, generate QR and card from it ---
        # This is the critical ordering: alloy is finalized before anything
        # references its hash. Card and QR are derived FROM the final hash.
        alloy_hash = ""
        if alloy_files:
            alloy_hash = hashlib.sha256(alloy_files[0].read_bytes()).hexdigest()
            self.log(f"  Alloy hash: {alloy_hash[:16]}")

        qr_path = self._generate_qr(ctx, repo_id)

        card = self._generate_card(ctx)
        card_path = ctx.output_dir / "README.md"
        card_path.write_text(card)
        self.log(f"  Card generated ({len(card)} chars)")

        # --- PHASE 4: Upload alloy, QR, card (all consistent) ---
        if alloy_files and include_alloy:
            api.upload_file(path_or_fileobj=str(alloy_files[0]),
                            path_in_repo=alloy_files[0].name, repo_id=repo_id)
            self.log(f"  Uploaded alloy: {alloy_files[0].name}")

        if qr_path and qr_path.exists():
            api.upload_file(path_or_fileobj=str(qr_path), path_in_repo="alloy-qr.png", repo_id=repo_id)

        api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md", repo_id=repo_id)

        self.log(f"  PUBLISHED: {pub_url}")
        self.log(f"  Verify: https://cambriantech.github.io/forge-alloy/verify/#{alloy_hash[:16]}")

        return ctx

    def _generate_card(self, ctx: ForgeContext) -> str:
        """Generate model card from alloy using alloy_to_card."""
        scripts_dir = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(scripts_dir))
        try:
            from alloy_to_card import alloy_to_card
            alloy_path = list(ctx.output_dir.glob("*.alloy.json"))
            alloy_hash = ""
            if alloy_path:
                alloy_hash = hashlib.sha256(alloy_path[0].read_bytes()).hexdigest()
            return alloy_to_card(ctx.alloy, alloy_hash)
        except ImportError:
            self.log("  WARNING: alloy_to_card.py not found — using basic card")
            return f"# {ctx.alloy.get('name', 'Model')}\n\nForged with ForgeAlloy.\n"

    def _verify_integrity(self, ctx: ForgeContext) -> list:
        """Verify alloy hashes match actual files."""
        errors = []
        integrity = (ctx.alloy.get("results") or {}).get("integrity", {})
        if not integrity:
            return []

        claimed_hash = integrity.get("modelHash", "")
        if claimed_hash:
            model_dir = ctx.output_dir / "model"
            safetensors = sorted(model_dir.glob("*.safetensors"))
            if safetensors:
                h = hashlib.sha256()
                for sf in safetensors:
                    with open(sf, 'rb') as f:
                        while True:
                            chunk = f.read(65536)
                            if not chunk:
                                break
                            h.update(chunk)
                actual = f"sha256:{h.hexdigest()}"
                if actual != claimed_hash:
                    errors.append(f"Model hash mismatch: claimed {claimed_hash[:24]}... actual {actual[:24]}...")
        return errors

    def _generate_qr(self, ctx: ForgeContext, repo_id: str) -> Path:
        """Generate QR code linking to verify page."""
        try:
            import qrcode
        except ImportError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install",
                                       "qrcode[pil]", "--quiet"])
                import qrcode
            except Exception:
                self.log("  QR generation skipped (install qrcode)")
                return None

        alloy_files = list(ctx.output_dir.glob("*.alloy.json"))
        if not alloy_files:
            return None

        alloy_hash = hashlib.sha256(alloy_files[0].read_bytes()).hexdigest()[:16]
        verify_url = f"https://cambriantech.github.io/forge-alloy/verify/#{alloy_hash}"

        qr = qrcode.make(verify_url)
        qr_path = ctx.output_dir / "alloy-qr.png"
        qr.save(str(qr_path))
        self.log(f"  QR → {verify_url}")
        return qr_path


class DeployExecutor(StageExecutor):
    """Deploy to a grid node for serving."""

    def execute(self, ctx: ForgeContext) -> ForgeContext:
        target = self.config.get("target", "local")
        health = self.config.get("healthCheck", True)
        warmup = self.config.get("warmup", True)
        concurrency = self.config.get("maxConcurrency", 4)
        auto_scale = self.config.get("autoScale", False)

        self.log(f"Deploy target: {target}")
        self.log(f"  Health check: {health}, Warmup: {warmup}")
        self.log(f"  Concurrency: {concurrency}, Auto-scale: {auto_scale}")

        if target == "local":
            self.log(f"  Model available at: {ctx.output_dir / 'model'}")
        elif target in ("bigmama", "grid"):
            self.log(f"  Grid deploy requires SSH/reticulum transport")
        else:
            self.log(f"  Custom target: {target}")

        return ctx
