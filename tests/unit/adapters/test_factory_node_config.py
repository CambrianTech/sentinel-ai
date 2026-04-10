"""TDD spec for factory_node.toml — declarative storage hierarchy.

Per-node config that overrides auto-detection. Lives at
~/sentinel-factory/.factory/factory_node.toml. Single source of truth
for which storage paths belong to which cache tier on this node.

The mental model is L0..L5 cache hierarchy:

  L0  GPU VRAM         volatile, microseconds, $$$$
  L1  System RAM       volatile, nanoseconds, $$$
  L2  Hot SSD          persistent, ~50µs, $$
  L3  Cold HDD         persistent, ~5ms, $
  L4  Network archive  persistent, seconds, $
  L5  HuggingFace      re-fetchable, infinite, free

factory_node.toml only describes L2+ (the persistent layers). L0 and
L1 are managed by torch/python directly. L5 is the fallback for
anything we evict that we might want back.

The grid (continuum) eventually reads this same file across all nodes
to make routing decisions: "don't push a Mixtral 8x22B forge to a
node whose L2 has only 500GB free; pick the node with the WD Red
Pro 16TB cold tier instead."

Schema:

  [node]
  name        = "bigmama"
  hostname    = "BigMama"
  roles       = ["forge"]            # forge | ship | both
  gpu_count   = 1
  gpu_vram_gb = 32

  [storage]
  hot = { path = "/home/joel/sentinel-factory/.factory", min_free_gb = 30 }

  [[storage.cold]]
  name        = "melmac"
  path        = "/mnt/d/cold"
  fs_type     = "drvfs"
  write_mb_per_sec = 210
  purpose     = ["work-archive", "published-backup"]

  [grid]
  coordinator = "tailscale://continuum-coordinator:7100"
  heartbeat_interval_seconds = 30

When the file doesn't exist, factory_storage falls back to auto-
detection (the current behavior). The two coexist gracefully:
declarative wins, auto-detect is the bootstrap fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


SAMPLE_TOML = """
[node]
name = "bigmama"
hostname = "BigMama"
roles = ["forge"]
gpu_count = 1
gpu_vram_gb = 32

[storage]
hot = { path = "/home/joel/sentinel-factory/.factory", min_free_gb = 30 }

[[storage.cold]]
name = "melmac"
path = "/mnt/d/cold"
fs_type = "drvfs"
write_mb_per_sec = 210
purpose = ["work-archive", "published-backup"]

[grid]
coordinator = "tailscale://continuum-coordinator:7100"
heartbeat_interval_seconds = 30
"""


# ── Loader ──────────────────────────────────────────────────────────────────


def test_factory_node_config_class_importable():
    from factory_storage import FactoryNodeConfig
    assert FactoryNodeConfig is not None


def test_loads_from_toml_file(tmp_path):
    from factory_storage import FactoryNodeConfig
    cfg_path = tmp_path / "factory_node.toml"
    cfg_path.write_text(SAMPLE_TOML)
    cfg = FactoryNodeConfig.from_file(cfg_path)
    assert cfg.node_name == "bigmama"
    assert cfg.hostname == "BigMama"
    assert cfg.gpu_count == 1
    assert cfg.gpu_vram_gb == 32
    assert "forge" in cfg.roles


def test_hot_tier_loaded(tmp_path):
    from factory_storage import FactoryNodeConfig
    cfg_path = tmp_path / "factory_node.toml"
    cfg_path.write_text(SAMPLE_TOML)
    cfg = FactoryNodeConfig.from_file(cfg_path)
    assert cfg.hot_path == Path("/home/joel/sentinel-factory/.factory")
    assert cfg.hot_min_free_gb == 30


def test_cold_tiers_loaded_in_order(tmp_path):
    from factory_storage import FactoryNodeConfig
    cfg_path = tmp_path / "factory_node.toml"
    cfg_path.write_text(SAMPLE_TOML)
    cfg = FactoryNodeConfig.from_file(cfg_path)
    assert len(cfg.cold_tiers) == 1
    assert cfg.cold_tiers[0].name == "melmac"
    assert cfg.cold_tiers[0].path == Path("/mnt/d/cold")
    assert cfg.cold_tiers[0].fs_type == "drvfs"
    assert cfg.cold_tiers[0].write_mb_per_sec == 210


def test_multiple_cold_tiers_in_priority_order(tmp_path):
    """Multi-cold-tier setup: list in spill order — fill tier 1 first,
    spill to tier 2 when tier 1 hits min_free, etc."""
    from factory_storage import FactoryNodeConfig
    cfg_path = tmp_path / "factory_node.toml"
    cfg_path.write_text("""
[node]
name = "multi-tier-test"
hostname = "test"

[storage]
hot = { path = "/tmp/hot" }

[[storage.cold]]
name = "fast-cold"
path = "/mnt/d/cold"

[[storage.cold]]
name = "archive"
path = "/mnt/e/cold"
""")
    cfg = FactoryNodeConfig.from_file(cfg_path)
    assert len(cfg.cold_tiers) == 2
    assert cfg.cold_tiers[0].name == "fast-cold"
    assert cfg.cold_tiers[1].name == "archive"


def test_missing_file_returns_none(tmp_path):
    from factory_storage import FactoryNodeConfig
    cfg_path = tmp_path / "factory_node.toml"
    cfg = FactoryNodeConfig.from_file(cfg_path)
    assert cfg is None


def test_invalid_toml_returns_none_with_warning(tmp_path, capsys):
    from factory_storage import FactoryNodeConfig
    cfg_path = tmp_path / "factory_node.toml"
    cfg_path.write_text("this is not valid toml [[[")
    cfg = FactoryNodeConfig.from_file(cfg_path)
    assert cfg is None


# ── Integration with auto_cleanup ───────────────────────────────────────────


def test_auto_cleanup_uses_first_cold_tier_when_config_exists(tmp_path):
    """When factory_node.toml declares cold tiers, auto_cleanup should
    use the first one as cold_root automatically. No need to pass
    --cleanup-cold-root on the daemon command line."""
    from factory_storage import auto_cleanup, FactoryNodeConfig
    from factory_queue import FactoryQueue

    # Set up a queue with an orphan work dir to evict
    q = FactoryQueue(tmp_path)
    work = tmp_path / "work" / "abandoned"
    work.mkdir(parents=True)
    (work / "model.safetensors").write_bytes(b"x" * 1024)

    # Set up cold tier path
    cold = tmp_path / "cold"
    cold.mkdir()

    # Write config that points cold tier here
    cfg_path = tmp_path / "factory_node.toml"
    cfg_path.write_text(f"""
[node]
name = "test"
hostname = "test"

[storage]
hot = {{ path = "{tmp_path}" }}

[[storage.cold]]
name = "primary-cold"
path = "{cold}"
""")

    # auto_cleanup with config_aware=True should auto-pick the cold tier
    report = auto_cleanup(tmp_path, force=True, config_aware=True)

    # Orphan should have moved to cold, not been deleted
    assert not (tmp_path / "work" / "abandoned").exists()
    assert (cold / "abandoned").exists(), "auto_cleanup should have moved to first cold tier from config"


def test_auto_cleanup_falls_back_to_explicit_cold_root_when_no_config(tmp_path):
    """No factory_node.toml present → auto_cleanup respects the
    explicit cold_root parameter (current behavior, backwards compat)."""
    from factory_storage import auto_cleanup
    from factory_queue import FactoryQueue
    q = FactoryQueue(tmp_path)
    work = tmp_path / "work" / "abandoned"
    work.mkdir(parents=True)
    (work / "model.safetensors").write_bytes(b"x" * 1024)
    cold = tmp_path / "explicit-cold"
    cold.mkdir()

    # No config file at tmp_path/factory_node.toml — should use explicit
    report = auto_cleanup(tmp_path, force=True, cold_root=cold, config_aware=True)
    assert (cold / "abandoned").exists()
