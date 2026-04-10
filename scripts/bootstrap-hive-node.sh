#!/usr/bin/env bash
# bootstrap-hive-node.sh — make a fresh box ready to be a forge grid node.
#
# What this script does (idempotent — safe to re-run after reboot or
# after a partial install):
#
#   1. Generates an ed25519 SSH key for git ↔ github auth (if missing)
#   2. Adds github.com to known_hosts (no first-connect prompt)
#   3. Persists HF_TOKEN + WSL nvidia-smi PATH to ~/.bashrc *before*
#      the non-interactive guard, so `ssh node 'cmd'` inherits them
#   4. Installs start-factory-daemon.sh as the one-command recovery
#      entrypoint
#   5. Verifies ssh.socket + tailscale autostart so the node comes back
#      online after every power-failure or fault
#   6. Prints the public key for the operator to register on github
#   7. Reports anything that needs manual attention
#
# Designed for the typical post-power-failure + drive-install +
# fresh-ubuntu + WSL2-on-Windows recovery scenarios. The principle:
# every step is idempotent, every step prints what it did, no step
# silently fails. If a node passes this script clean, it can be
# remote-controlled from FlashGordon (or any operator box) without
# any further interactive setup.
#
# Usage:
#
#   curl -sSf https://raw.githubusercontent.com/CambrianTech/sentinel-ai/main/scripts/bootstrap-hive-node.sh | HF_TOKEN=hf_xxx bash
#
#   OR clone the repo first, then:
#
#   HF_TOKEN=hf_xxx ./scripts/bootstrap-hive-node.sh
#
# Requirements:
#   - bash, ssh-keygen, ssh-keyscan, python3
#   - HF_TOKEN env var set if you want HF publish to work
#   - The repo cloned somewhere (the script assumes ~/sentinel-ai or
#     ~/sentinel-factory; falls back to current dir)

set -euo pipefail

cyan()  { printf '\033[36m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
red()   { printf '\033[31m%s\033[0m\n' "$*"; }
yellow(){ printf '\033[33m%s\033[0m\n' "$*"; }

cyan "═══ hive node bootstrap ═══"
echo "  host:  $(hostname)"
echo "  user:  $(whoami)"
echo "  os:    $(uname -sr)"
[ -f /proc/version ] && grep -q microsoft /proc/version && echo "  wsl:   $(grep microsoft /proc/version | head -1)"
echo ""

# ── 1. SSH key for git ──────────────────────────────────────────────────────
cyan "[1/7] git SSH key"
mkdir -p ~/.ssh
chmod 700 ~/.ssh
if [ -f ~/.ssh/id_ed25519 ]; then
    green "  existing ~/.ssh/id_ed25519 — keeping"
else
    ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "$(whoami)@$(hostname)" 2>&1 | tail -3
    green "  generated new ed25519 keypair"
fi
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub

# ── 2. github.com known_hosts ───────────────────────────────────────────────
cyan "[2/7] known_hosts"
if ssh-keygen -F github.com -f ~/.ssh/known_hosts > /dev/null 2>&1; then
    green "  github.com already in known_hosts"
else
    ssh-keyscan -t ed25519,rsa github.com 2>/dev/null >> ~/.ssh/known_hosts
    green "  added github.com fingerprint"
fi
chmod 644 ~/.ssh/known_hosts

# ── 3. ~/.bashrc env prelude (persistent across reboots + non-interactive ssh) ──
cyan "[3/7] bashrc env prelude"
PRELUDE_MARK="auto-recovery (bootstrap-hive-node)"
if grep -q "$PRELUDE_MARK" ~/.bashrc 2>/dev/null; then
    green "  prelude already in ~/.bashrc"
else
    [ -f ~/.bashrc ] || touch ~/.bashrc
    EXISTING=$(cat ~/.bashrc)
    {
        echo "# === $PRELUDE_MARK ==="
        echo "# Set BEFORE the non-interactive early-exit so that"
        echo "# 'ssh node command' inherits these vars (the script-execution"
        echo "# code path that the grid uses to remote-control nodes)."
        if [ -n "${HF_TOKEN:-}" ]; then
            echo "export HF_TOKEN=$HF_TOKEN"
        else
            echo "# HF_TOKEN: not provided at bootstrap time. Set later via:"
            echo "#   echo 'export HF_TOKEN=hf_xxx' >> ~/.bashrc"
            echo "# or re-run this script with HF_TOKEN=hf_xxx in env."
        fi
        echo "# WSL2 on Windows ships nvidia-smi at /usr/lib/wsl/lib but doesn't"
        echo "# add it to PATH. Without this line, every command that needs"
        echo "# nvidia-smi has to set PATH manually."
        if [ -d /usr/lib/wsl/lib ]; then
            echo "export PATH=/usr/lib/wsl/lib:\$PATH"
        fi
        echo "# === end $PRELUDE_MARK ==="
        echo ""
        echo "$EXISTING"
    } > ~/.bashrc
    green "  prepended prelude to ~/.bashrc"
fi

# ── 3.5. factory_node.toml.example template ────────────────────────────────
cyan "[3.5/7] factory_node.toml.example template"
TEMPLATE=~/factory_node.toml.example
if [ -f "$TEMPLATE" ]; then
    green "  $TEMPLATE already exists — keeping"
else
    cat > "$TEMPLATE" <<TEMPLATE_EOF
# factory_node.toml — declarative storage hierarchy for this hive node.
#
# Place this at <queue_root>/factory_node.toml (typically
# ~/sentinel-factory/.factory/factory_node.toml). The factory_queue
# daemon reads it on startup; declarative config wins over auto-detect.
#
# The grid (continuum) eventually reads this same file across all nodes
# to make routing decisions about which node forges which alloy.

[node]
name        = "$(hostname)"          # display name in heartbeats / grid view
hostname    = "$(hostname)"          # actual host (for grid coordination)
roles       = ["forge"]              # forge | ship | both
gpu_count   = 1                      # adjust per box
gpu_vram_gb = 32                     # 5090=32, 4090=24, 3090=24, 4060=8

[storage]
# Hot tier: fast SSD where the .factory/ queue dir lives.
# min_free_gb is a soft floor — auto-cleanup tries to keep this much free.
hot = { path = "$HOME/sentinel-factory/.factory", min_free_gb = 30 }

# Cold tiers in PRIORITY ORDER. Evictions fill tier 1 first, spill to
# tier 2 when tier 1 is full, etc. List as many as you have.
[[storage.cold]]
name             = "primary-cold"
path             = "/mnt/d/cold"     # adjust to your mount point
fs_type          = "drvfs"           # ext4 | drvfs | nfs | s3
write_mb_per_sec = 210               # informs the grid scheduler
purpose          = ["work-archive", "published-backup"]

# [[storage.cold]]
# name = "secondary-archive"
# path = "/mnt/e/cold"
# fs_type = "ext4"

[grid]
# Future — populated when continuum's grid coordinator is online.
# coordinator = "tailscale://continuum-coordinator:7100"
# heartbeat_interval_seconds = 30
TEMPLATE_EOF
    green "  installed $TEMPLATE"
    yellow "  to activate: cp $TEMPLATE ~/sentinel-factory/.factory/factory_node.toml"
    yellow "               (then edit the paths to match your hardware)"
fi

# ── 4. start-factory-daemon.sh wrapper ──────────────────────────────────────
cyan "[4/7] factory daemon recovery wrapper"
WRAPPER=~/start-factory-daemon.sh
cat > $WRAPPER <<'WRAPPER_EOF'
#!/usr/bin/env bash
# Post-reboot recovery: one command to bring this hive node back online.
# Sources env explicitly (works from cron/systemd where ~/.bashrc isn't
# sourced), pulls latest code, auto-detects cold drive, starts the
# factory_queue daemon via nohup, self-validates.
#
# Usage:
#   ssh bigmama "~/start-factory-daemon.sh"
#   ssh bigmama "~/start-factory-daemon.sh --max-iters 1"   # smoke test
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:-$(grep '^export HF_TOKEN' ~/.bashrc 2>/dev/null | head -1 | sed 's/export HF_TOKEN=//')}"
export PATH="/usr/lib/wsl/lib:${PATH}"

# Find the sentinel-ai checkout — try the worktree first, fall back to main
for dir in "$HOME/sentinel-factory" "$HOME/sentinel-ai" "$PWD"; do
    if [ -d "$dir/.git" ] || [ -f "$dir/.git" ]; then
        REPO="$dir"
        break
    fi
done
[ -z "${REPO:-}" ] && { echo "ERROR: no sentinel-ai checkout found"; exit 1; }
cd "$REPO"
echo "  using repo: $REPO"

# Resolve forge-alloy worktree if present (provides the schema fixes)
for fa in "$HOME/forge-alloy-domain" "$HOME/forge-alloy"; do
    if [ -d "$fa/python/forge_alloy" ]; then
        export PYTHONPATH="$fa/python:${PYTHONPATH:-}"
        echo "  forge-alloy: $fa"
        break
    fi
done

echo "=== git pull ==="
git pull --ff-only 2>&1 | tail -3 || echo "  (pull skipped — divergent or detached)"

echo ""
echo "=== sanity ==="
echo "  HF_TOKEN:    $([ -n "$HF_TOKEN" ] && echo set || echo MISSING)"
echo "  nvidia-smi:  $(which nvidia-smi 2>/dev/null || echo MISSING)"
echo "  python:      $(./.venv/bin/python --version 2>/dev/null || echo MISSING)"

# Detect cold drive at any of the conventional mount points
COLD_FLAG=""
for mnt in /mnt/cold /mnt/spinner /mnt/wd-red-pro /mnt/coldtier; do
    if [ -d "$mnt" ] && mountpoint -q "$mnt" 2>/dev/null; then
        COLD_FLAG="--cleanup-cold-root $mnt"
        echo "  cold drive:  $mnt (mounted)"
        break
    fi
done
[ -z "$COLD_FLAG" ] && echo "  cold drive:  NOT MOUNTED (forges may hit disk pressure)"

echo ""
echo "=== launching daemon ==="
mkdir -p .factory/line
nohup ./.venv/bin/python scripts/factory_queue.py \
    --root .factory \
    --idle-sleep 30 \
    --cleanup-threshold 90 \
    $COLD_FLAG \
    "$@" \
    > .factory/line/daemon.log 2>&1 < /dev/null &
PID=$!
disown
sleep 2
if kill -0 $PID 2>/dev/null; then
    echo "  daemon started, pid $PID"
    echo "  log:    .factory/line/daemon.log"
    echo "  status: ssh \$(hostname) \"$REPO/.venv/bin/python $REPO/scripts/factory_queue.py --root $REPO/.factory --status --pretty\""
else
    echo "  ERROR: daemon failed to start. tail -20 .factory/line/daemon.log:"
    tail -20 .factory/line/daemon.log
    exit 1
fi
WRAPPER_EOF
chmod +x $WRAPPER
green "  installed $WRAPPER"

# ── 5. Verify autostart services ────────────────────────────────────────────
cyan "[5/7] autostart services"
if command -v systemctl >/dev/null 2>&1; then
    SSH_SOCKET=$(systemctl is-enabled ssh.socket 2>/dev/null || echo "not-found")
    SSH_SERVICE=$(systemctl is-enabled ssh.service 2>/dev/null || echo "not-found")
    TS=$(systemctl is-enabled tailscaled 2>/dev/null || echo "not-found")
    [ "$SSH_SOCKET" = "enabled" ] && green "  ssh.socket:   enabled (port 22 will listen on boot via socket activation)"
    [ "$SSH_SOCKET" != "enabled" ] && [ "$SSH_SERVICE" = "enabled" ] && green "  ssh.service:  enabled"
    [ "$SSH_SOCKET" != "enabled" ] && [ "$SSH_SERVICE" != "enabled" ] && yellow "  ssh:          NOT auto-starting — run: sudo systemctl enable ssh.socket"
    [ "$TS" = "enabled" ] && green "  tailscaled:   enabled (grid mesh comes back online on boot)"
    [ "$TS" != "enabled" ] && yellow "  tailscaled:   NOT auto-starting — run: sudo systemctl enable tailscaled"
else
    yellow "  systemctl not available — autostart cannot be verified"
fi

# ── 6. Print public key for the operator ──────────────────────────────────────
cyan "[6/7] public key (paste into github.com → Settings → SSH keys)"
echo ""
yellow "  $(cat ~/.ssh/id_ed25519.pub)"
echo ""

# ── 7. Validate connectivity if the key is already registered ──────────────
cyan "[7/7] github auth test"
GH_RESPONSE=$(ssh -T -o StrictHostKeyChecking=accept-new -o BatchMode=yes git@github.com 2>&1 || true)
if echo "$GH_RESPONSE" | grep -q "successfully authenticated"; then
    green "  $(echo $GH_RESPONSE | head -1)"
else
    yellow "  not yet authenticated — register the public key above and re-run this script"
fi

echo ""
cyan "═══ bootstrap complete ═══"
echo ""
echo "Next steps:"
echo "  1. Paste the public key into github.com → Settings → SSH keys"
echo "  2. Re-run this script to verify auth works"
echo "  3. Mount the cold drive (if you have one) at /mnt/cold"
echo "  4. ~/start-factory-daemon.sh"
echo ""
echo "Auto-recovery test: shut down → boot back up → ssh \$(hostname) '~/start-factory-daemon.sh'"
echo "If that one command brings the node back online, the bootstrap worked."
