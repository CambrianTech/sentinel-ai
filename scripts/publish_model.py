#!/usr/bin/env python3
"""
publish_model.py — Publish a forged model to HuggingFace.

Standalone script that operates on files, no ForgeContext needed.
Reads the alloy + model weights from a forge output directory,
verifies integrity, generates card + QR, and uploads everything.

Critical ordering: finalize alloy → hash → QR → card → upload
(all derived from the same final hash)

Usage:
    python scripts/publish_model.py output/forged/qwen3.5-4b-code-128k-final/
    python scripts/publish_model.py output/forged/model-name/ --org continuum-ai
    python scripts/publish_model.py output/forged/model-name/ --dry-run
    python scripts/publish_model.py output/forged/model-name/ --json
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def hash_file(path: Path) -> str:
    """SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_model_weights(model_dir: Path) -> str:
    """SHA-256 hash of all safetensors files in order."""
    safetensors = sorted(model_dir.glob("*.safetensors"))
    if not safetensors:
        return ""
    h = hashlib.sha256()
    for sf in safetensors:
        with open(sf, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def verify_integrity(alloy: dict, model_dir: Path) -> list[str]:
    """Verify alloy claims match actual files. Returns list of errors."""
    errors = []
    integrity = alloy.get("results", {}).get("integrity", {})
    if not integrity:
        return []

    claimed_hash = integrity.get("modelHash", "")
    if claimed_hash:
        actual_hash = hash_model_weights(model_dir)
        if actual_hash and actual_hash != claimed_hash:
            errors.append(
                f"Model hash mismatch: claimed {claimed_hash[:32]}... "
                f"actual {actual_hash[:32]}..."
            )
    return errors


def generate_qr(verify_url: str, output_path: Path) -> bool:
    """Generate QR code image. Returns True on success."""
    try:
        import qrcode
    except ImportError:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "qrcode[pil]", "--quiet"],
                stdout=subprocess.DEVNULL,
            )
            import qrcode
        except Exception:
            print("  QR generation skipped (install qrcode[pil])")
            return False

    qr = qrcode.QRCode(version=1, box_size=10, border=2)
    qr.add_data(verify_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(str(output_path))
    return True


def generate_card(alloy: dict, alloy_hash: str) -> str:
    """Generate model card from alloy using alloy_to_card."""
    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    from alloy_to_card import alloy_to_card
    return alloy_to_card(alloy, alloy_hash)


def publish(output_dir: Path, org: str = "continuum-ai",
            repo_name: str = "", private: bool = False,
            dry_run: bool = False, json_output: bool = False) -> dict:
    """Publish a forge output directory to HuggingFace.

    Returns dict with repoUrl, verifyUrl, alloyHash, filesUploaded, totalSizeGb.
    """
    output_dir = output_dir.resolve()
    model_dir = output_dir / "model"

    # Find alloy
    alloy_files = list(output_dir.glob("*.alloy.json"))
    if not alloy_files:
        raise FileNotFoundError(f"No .alloy.json found in {output_dir}")
    alloy_path = alloy_files[0]
    alloy = json.loads(alloy_path.read_text())

    name = repo_name or alloy.get("name", alloy_path.stem.replace(".alloy", ""))
    repo_id = f"{org}/{name}"
    pub_url = f"https://huggingface.co/{repo_id}"
    pub_time = datetime.now(timezone.utc).isoformat()

    print(f"Publishing: {repo_id}")
    print(f"  Output dir: {output_dir}")
    print(f"  Alloy: {alloy_path.name}")

    # --- VERIFY INTEGRITY ---
    if model_dir.exists():
        errors = verify_integrity(alloy, model_dir)
        if errors:
            for err in errors:
                print(f"  INTEGRITY ERROR: {err}")
            raise RuntimeError("Integrity verification failed — aborting publish")
        print("  Integrity: OK")
    else:
        print("  WARNING: No model/ directory — publishing metadata only")

    # --- PHASE 1: FINALIZE ALLOY (write receipt, then hash) ---
    # Inject benchmark resultHashes from on-disk sample files BEFORE the
    # alloy is canonicalized and hashed. Each benchmark in
    # results.benchmarks may carry a `samplesPath` (relative to output_dir)
    # pointing at the per-problem evaluation output (e.g. evalplus's
    # sanitized JSONL). At publish time we compute SHA-256 of that file
    # and inject it as `resultHash`, which moves the benchmark from
    # "self-reported" to "attested" on the verify page — a third party can
    # download the JSONL from the same HF repo, recompute the hash, verify
    # it matches the alloy, and re-score against the per-problem outputs
    # without trusting the producer's claim. This is the cheapest concrete
    # trust upgrade per the forge-alloy attestation roadmap.
    bench_hashed = 0
    for b in alloy.get("results", {}).get("benchmarks", []):
        samples_rel = b.get("samplesPath")
        if not samples_rel or b.get("resultHash"):
            continue
        samples_abs = (output_dir / samples_rel).resolve()
        if not samples_abs.exists():
            print(f"  WARNING: benchmark '{b.get('name','?')}' samplesPath {samples_rel} not found — skipping resultHash")
            continue
        if not str(samples_abs).startswith(str(output_dir.resolve())):
            print(f"  WARNING: benchmark '{b.get('name','?')}' samplesPath {samples_rel} escapes publish dir — skipping")
            continue
        b["resultHash"] = "sha256:" + hash_file(samples_abs)
        bench_hashed += 1
    if bench_hashed:
        print(f"  Result-hashed {bench_hashed} benchmark sample file(s)")

    alloy["receipt"] = {
        "publications": [{
            "target": "huggingface",
            "url": pub_url,
            "publishedAt": pub_time,
        }],
        "issuedAt": pub_time,
    }
    # Ensure author is set
    if not alloy.get("author"):
        alloy["author"] = org

    alloy_json = json.dumps(alloy, indent=2)
    alloy_hash = hashlib.sha256(alloy_json.encode()).hexdigest()
    if not dry_run:
        alloy_path.write_text(alloy_json)
    print(f"  Alloy hash: {alloy_hash[:16]}")

    # --- PHASE 2: GENERATE QR + CARD (from final hash) ---
    verify_url = f"https://cambriantech.github.io/forge-alloy/verify/#{alloy_hash[:16]}"

    qr_path = output_dir / "alloy-qr.png"
    qr_ok = False
    if not dry_run:
        qr_ok = generate_qr(verify_url, qr_path)
    print(f"  QR: {verify_url}")

    card = generate_card(alloy, alloy_hash)
    card_path = output_dir / "README.md"
    if not dry_run:
        card_path.write_text(card)
    print(f"  Card: {len(card)} chars")

    if dry_run:
        print("\n  DRY RUN — not uploading. Files prepared in output dir.")
        result = {
            "success": True,
            "dryRun": True,
            "repoUrl": pub_url,
            "verifyUrl": verify_url,
            "alloyHash": alloy_hash,
            "cardPath": str(card_path),
        }
        if json_output:
            print(json.dumps(result, indent=2))
        return result

    # --- PHASE 3: UPLOAD TO HUGGINGFACE ---
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("  ERROR: huggingface_hub not installed")
        print("  Install: pip install huggingface_hub")
        raise

    api = HfApi()
    files_uploaded = 0
    total_bytes = 0

    # Create repo
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
        print(f"  Repo: {repo_id}")
    except Exception as e:
        raise RuntimeError(f"Failed to create repo: {e}")

    # Upload model weights
    if model_dir.exists():
        safetensors = list(model_dir.glob("*.safetensors"))
        if safetensors:
            print(f"  Uploading {len(safetensors)} weight files...")
            for sf in safetensors:
                api.upload_file(path_or_fileobj=str(sf), path_in_repo=sf.name, repo_id=repo_id)
                total_bytes += sf.stat().st_size
                files_uploaded += 1

        # Config files
        for cfg_name in ["config.json", "tokenizer.json", "tokenizer_config.json",
                         "generation_config.json", "special_tokens_map.json",
                         "chat_template.jinja"]:
            cfg_path = model_dir / cfg_name
            if cfg_path.exists():
                api.upload_file(path_or_fileobj=str(cfg_path), path_in_repo=cfg_name, repo_id=repo_id)
                files_uploaded += 1

    # Upload benchmark samples
    bench_dir = output_dir / "benchmark"
    if bench_dir.exists():
        for txt in bench_dir.glob("*"):
            if txt.is_file():
                api.upload_file(path_or_fileobj=str(txt),
                                path_in_repo=f"benchmark/{txt.name}", repo_id=repo_id)
                files_uploaded += 1

    # Upload eval results
    eval_dir = output_dir / "eval"
    if eval_dir.exists():
        for f in eval_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(output_dir)
                api.upload_file(path_or_fileobj=str(f), path_in_repo=str(rel), repo_id=repo_id)
                files_uploaded += 1

    # Upload alloy (finalized with receipt)
    api.upload_file(path_or_fileobj=str(alloy_path), path_in_repo=alloy_path.name, repo_id=repo_id)
    files_uploaded += 1

    # Upload QR
    if qr_ok and qr_path.exists():
        api.upload_file(path_or_fileobj=str(qr_path), path_in_repo="alloy-qr.png", repo_id=repo_id)
        files_uploaded += 1

    # Upload card LAST (references alloy hash which is already uploaded)
    api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md", repo_id=repo_id)
    files_uploaded += 1

    total_gb = total_bytes / (1024 ** 3)

    print(f"\n  PUBLISHED: {pub_url}")
    print(f"  Verify: {verify_url}")
    print(f"  Files: {files_uploaded}, Size: {total_gb:.1f}GB")

    result = {
        "success": True,
        "repoUrl": pub_url,
        "verifyUrl": verify_url,
        "alloyHash": alloy_hash,
        "modelHash": hash_model_weights(model_dir) if model_dir.exists() else "",
        "filesUploaded": files_uploaded,
        "totalSizeGb": round(total_gb, 2),
    }

    if json_output:
        print(json.dumps(result, indent=2))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Publish a forged model to HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/publish_model.py output/forged/qwen3.5-4b-code-128k-final/
  python scripts/publish_model.py output/forged/model/ --org continuum-ai --dry-run
  python scripts/publish_model.py output/forged/model/ --json
        """,
    )
    parser.add_argument("output_dir", type=Path, help="Forge output directory")
    parser.add_argument("--org", default="continuum-ai", help="HuggingFace org (default: continuum-ai)")
    parser.add_argument("--repo-name", default="", help="Override repo name (default: from alloy)")
    parser.add_argument("--private", action="store_true", help="Publish as private repo")
    parser.add_argument("--dry-run", action="store_true", help="Prepare files without uploading")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    args = parser.parse_args()

    if not args.output_dir.exists():
        print(f"ERROR: {args.output_dir} does not exist")
        sys.exit(1)

    try:
        publish(
            output_dir=args.output_dir,
            org=args.org,
            repo_name=args.repo_name,
            private=args.private,
            dry_run=args.dry_run,
            json_output=args.json,
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
