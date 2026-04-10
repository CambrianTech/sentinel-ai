# Foundry Filesystem Setup

**Status**: Requirement. Every sentinel-ai forge node MUST have its source-weight cache on a native Linux filesystem (xfs preferred, ext4 acceptable). **drvfs / 9p / ntfs-3g / CIFS are forbidden for source-weight reads** — they will silently wedge mid-forge on big-MoE models. This document is the operator playbook for doing it right.

## TL;DR

If you are running a forge node on WSL2 (the common case for BigMama-class boxes with a Windows host):

1. **Reformat your cold-tier drive as xfs natively in WSL2.** Do NOT keep it NTFS and mount via drvfs.
2. **Symlink `~/.cache/huggingface` to the xfs mount** so every HF download goes to the cold tier automatically.
3. **Validate** that the mount survives a `wsl --shutdown` and that the daemon can read big safetensors sustained without hanging.

The exact commands for each step are below. Total time: ~90 minutes including re-downloading your model cache. **The alternative is that your forge wedges at ~70% of a big-MoE load and you can't recover without restarting WSL from the Windows side.** We know because we lost several hours to this failure mode ourselves.

---

## Why this matters — the drvfs hang

### What happened

A sentinel-ai forge node on BigMama was attempting to load Mixtral 8x7B (93 GB fp16, 19 safetensors shards) via Accelerate's streaming device-map load. The source weights were sitting in the HuggingFace cache, which had been symlinked to `/mnt/d/cold/huggingface-cache` — a Windows-mounted NTFS drive accessed through WSL2's drvfs layer (which uses the 9p protocol internally).

The load progressed normally to shard 207 of 291, and then wedged. The daemon stopped making progress. The heartbeat file stopped updating. The process entered the `D` state (uninterruptible kernel sleep). `SIGTERM` and `SIGKILL` had no effect — processes in `D` state cannot be killed from userspace.

### The diagnosis

Reading `/proc/$PID/wchan` for the wedged thread showed:

```
wchan = p9_client_rpc
```

The main thread was blocked inside the 9p client RPC layer — WSL2's mechanism for talking to Windows-mounted filesystems — waiting for a response from the Windows-side drvfs service that never came. The 9p protocol didn't error out; it simply stopped responding. No timeout. No kernel log entry. No recovery path.

Every other thread in the 76-thread process was blocked in `futex_wait_queue`, which is the state threads enter when they're waiting for the GIL — and the main thread held the GIL inside the blocked C extension. **Not even the heartbeat thread could run**, because the heartbeat thread is a Python thread and needs the GIL to execute, and the GIL was held by the main thread that was blocked in the kernel.

The only recovery was `wsl --shutdown` from a Windows PowerShell, which killed WSL2 as a whole and freed the drvfs layer.

### Why drvfs is unsuitable

drvfs / 9p is fine for small files and occasional access. It was designed for user convenience — being able to `cd /mnt/c` from WSL2 and interact with Windows files. **It was not designed for sustained sequential reads of multi-gigabyte files**, which is exactly what a forge pipeline does when it loads a big-MoE source model.

The failure mode is the worst possible kind:

1. **Silent** — no error message, no kernel log entry, no timeout. Just a stopped process.
2. **Unkillable from userspace** — the `D` state means `kill -9` won't work; only WSL-level shutdown frees it.
3. **Probabilistic** — sometimes you get through 71% of a load, sometimes 95%, sometimes the whole thing. You can't reproduce it reliably, which means you can't protect against it with retries.
4. **Corrupts nothing but wastes everything** — the data on disk is fine, but every minute of forge time spent before the wedge is lost.

**Native Linux filesystems (xfs, ext4, btrfs) don't have this failure mode** because there's no 9p RPC layer between the kernel and the disk. The kernel reads from the disk directly, the way kernels are designed to read from disks.

### Why xfs specifically

- **Designed for big sequential files.** Originally built at SGI for film and scientific data workloads — petabyte-scale media stores reading huge sequential files. The forge workload is *exactly* what xfs was built to win on.
- **Faster than ext4 on big-file sequential reads** by a measurable margin in every benchmark we've seen.
- **Faster than btrfs on HDD workloads** because btrfs's CoW design fragments large files over time; xfs's extent-based allocation keeps files contiguous.
- **Mature and boring.** Default on RHEL 7+. Used by every HPC site. Used by film studios. The last interesting bug report was years ago.
- **Native in the WSL2 kernel.** No DKMS module, no FUSE layer, no compile step.
- **Journaled**, so it survives power loss cleanly without manual fsck.
- **`reflink` support** for copy-on-write file clones if you ever want to snapshot a model directory before forging into it (`cp --reflink=always`).

**ext4 is an acceptable fallback** if xfs is unavailable for any reason — it works, it's boring, it's slightly slower on big files and slightly faster on small files. For the forge workload the difference is measurable but not decisive.

---

## Setup: WSL2 host with Windows-attached drive

This is the setup we walked through for BigMama on 2026-04-10. It assumes:

- Windows 11 (or Windows 10 with WSL 2.0+)
- An installed WSL2 distribution (Ubuntu, Debian, etc.)
- A dedicated data drive (SATA HDD or SSD) currently formatted NTFS or unformatted
- PowerShell access to the Windows host as Administrator
- The daemon is not currently running on the WSL2 side

### Step 1 — Identify the drive

In a **Windows PowerShell as Administrator**:

```powershell
# List all physical disks with size and bus type so you can identify the target
Get-Disk | Select-Object Number, FriendlyName, @{Name="SizeGB";Expression={[math]::Round($_.Size/1GB,0)}}, BusType

# If the target drive has a Windows letter (like D:), look up its disk number
(Get-Partition -DriveLetter D).DiskNumber
```

Match the disk number to the drive you want to reformat. **Confirm you have the right drive before proceeding** — the next steps wipe it.

### Step 2 — Take the drive offline in Windows

```powershell
# Replace <N> with the disk number from Step 1
Set-Disk -Number <N> -IsOffline $true
```

This releases the drive from Windows so WSL2 can claim exclusive access. The Windows drive letter disappears immediately. The data on disk is not touched by this command (it just marks the disk offline from Windows' perspective), so you can revert with `Set-Disk -Number <N> -IsOffline $false` if you need to back out.

### Step 3 — Mount the drive into WSL2 as a raw block device

```powershell
# Still in elevated PowerShell
wsl --mount \\.\PHYSICALDRIVE<N> --bare
```

The `--bare` flag is critical. It tells WSL2 "give me the raw block device, do not try to interpret the partition table or auto-mount anything." Without `--bare`, WSL2 would look at the existing NTFS partition and try to mount it via its built-in ntfs3 driver, which we do not want — we are reformatting.

### Step 4 — Confirm the drive in WSL2 and install xfsprogs

Open a WSL terminal (or use an existing session). Confirm the new block device:

```bash
lsblk
```

You should see a new disk — typically `/dev/sdb`, `/dev/sdc`, `/dev/sdg`, etc., depending on what else is plugged in — with its full size. Existing NTFS partitions show as children of the disk (we will wipe them in the next step).

Install xfsprogs if it is not already present:

```bash
sudo apt-get update
sudo apt-get install -y xfsprogs
```

Ignore any `systemd is not running` warnings — xfsprogs does not need systemd for the binary you actually care about (`mkfs.xfs`). Verify the install:

```bash
which mkfs.xfs && mkfs.xfs -V
```

Should print a path like `/usr/sbin/mkfs.xfs` and a version like `mkfs.xfs version 6.6.0`.

### Step 5 — Wipe and format the drive

```bash
# Replace /dev/sdg with the device name lsblk reported in Step 4
# Safety check — confirm nothing is mounted
mount | grep sdg

# Wipe all filesystem signatures and the GPT partition table
sudo wipefs -a /dev/sdg

# Format as xfs with tuned allocation groups for big-file sequential I/O
sudo mkfs.xfs -f -L cold -d agcount=16 /dev/sdg
```

The `-d agcount=16` option sets the number of allocation groups to 16 (the default on a single-disk filesystem this size would be 4). Sixteen allocation groups is the sweet spot for big-file sequential I/O on a 16 TB HDD — it lets the kernel parallelize allocation across the drive without spreading any single big file across too many groups. If your drive is smaller, scale proportionally: 4 AGs per TB is a reasonable rule of thumb, with 4 as the minimum and 32 as the maximum.

The `-L cold` sets a filesystem label so you can use `LABEL=cold` in `/etc/fstab` and never worry about the device name changing if you ever plug in or unplug other drives.

### Step 6 — Mount the new filesystem

```bash
sudo mkdir -p /mnt/cold
sudo mount -t xfs -o noatime LABEL=cold /mnt/cold
sudo chown -R $USER:$USER /mnt/cold
```

The `noatime` option tells the kernel not to update access timestamps on every read, which is a meaningful write-amplification reduction when you are sequentially reading 100+ GB of safetensors shards. The rest of xfs's default mount options are sensible and do not need changing.

Verify:

```bash
df -h /mnt/cold
xfs_info /mnt/cold
```

You should see the full disk size available and `agcount=16` in the xfs_info output.

### Step 7 — Symlink the HuggingFace cache to the new mount

Every forge node needs the HF cache on the cold tier so `snapshot_download` and `from_pretrained` automatically use it:

```bash
# Remove any existing cache (or symlink) in the default location
rm -f ~/.cache/huggingface

# Create the new cache directory on the cold tier
mkdir -p /mnt/cold/huggingface-cache

# Symlink the default location to the cold tier
ln -s /mnt/cold/huggingface-cache ~/.cache/huggingface

# Verify
ls -la ~/.cache/huggingface
touch ~/.cache/huggingface/.test-write && rm ~/.cache/huggingface/.test-write && echo "write OK"
```

If you had an existing HF cache elsewhere with downloaded models, copy them into the new location before removing the old cache. For most operators the simpler move is to just let HuggingFace re-download on demand — at multi-Gbit bandwidth a 93 GB model takes under 10 minutes, and the alternative is spending hours copying through drvfs which is slow anyway.

### Step 8 — Persist the mount across `wsl --shutdown`

WSL2 does not remember the `wsl --mount --bare` call across shutdown. The cleanest workaround today is a small bootstrap script that re-runs the mount and the block-device discovery on each WSL startup. Add to `~/.bashrc` or as a systemd unit if you have systemd enabled in WSL2:

```bash
# ~/.config/foundry/mount-cold.sh
#!/bin/bash
# Re-mount the cold tier xfs if it's not already mounted
if ! mountpoint -q /mnt/cold; then
    # Find the block device by label
    DEVICE=$(blkid -L cold 2>/dev/null)
    if [ -n "$DEVICE" ]; then
        sudo mount -t xfs -o noatime "$DEVICE" /mnt/cold
    fi
fi
```

And on the Windows side, you will need to re-run `wsl --mount \\.\PHYSICALDRIVE<N> --bare` after any `wsl --shutdown`. This is a known WSL2 limitation; the Foreman role (see `continuum/docs/` when it lands) will eventually automate this step.

For now, the manual post-shutdown sequence is:

```powershell
# In elevated PowerShell after wsl --shutdown
wsl --mount \\.\PHYSICALDRIVE<N> --bare
```

Then inside WSL, the bootstrap script (or a manual `sudo mount -t xfs -o noatime LABEL=cold /mnt/cold`) remounts the filesystem.

---

## Setup: native Linux host with directly-attached drive

Much simpler — no WSL layer, no drvfs, no Windows-side steps. Skip to step 4 of the WSL2 setup, use `lsblk` to identify your drive, and run steps 5, 6, and 7 directly. Add the mount to `/etc/fstab` for automatic mounting across reboots:

```
LABEL=cold  /mnt/cold  xfs  defaults,noatime  0  0
```

That is the entire setup for a native Linux foundry node. No `wsl --mount`, no `wsl --shutdown` rediscovery dance. If you have a choice between running your foundry on WSL2 or on native Linux, **native Linux is simpler by one full layer of operational complexity.**

---

## Setup: network-attached storage (NFS, iSCSI, SMB)

Short version: **don't**, unless you know exactly why you're doing it and you're prepared to handle the failure modes.

Network filesystems can work, but they introduce a new class of failure modes (network partitions, server crashes, RPC timeouts) that the forge pipeline is not designed to handle. If your source weights live on an NFS server and the server goes down mid-forge, you will see symptoms similar to the drvfs hang — blocked processes in `D` state that can't be killed cleanly. The fact that NFS hangs are more diagnosable than drvfs hangs (you can see the NFS client state in `/proc/self/mountstats`) is a small consolation; you still lose the forge.

If you must use network storage, the rules:

- **Use NFS v4 with `soft` mount options**, not `hard`. A `hard` mount blocks indefinitely on server loss; a `soft` mount returns an error after a timeout, which the forge can handle.
- **Run a local cache in front of the network filesystem** using something like `cachefilesd` or manually rsyncing to a local xfs mount before forging.
- **Monitor the server-side network filesystem as aggressively as you monitor the forge itself.** A mid-forge network partition is much more likely than a mid-forge kernel bug.

The cleanest answer is: treat network storage as the transport for *forged artifacts* (write-once, read-rarely, which can tolerate latency), and keep source weights on a local xfs mount. This matches the `[[storage.cold]]` and `[[storage.network]]` tier separation in `docs/FACTORY-PROTOCOL.md`.

---

## Validation: confirm the setup works before committing to it

Before queueing a big forge on your new foundry, run the validation sequence to confirm the cold tier actually works for sustained sequential reads:

```bash
# 1. Write throughput — baseline
dd if=/dev/zero of=/mnt/cold/.test.bin bs=1M count=1024 conv=fdatasync status=progress
# Should show 200+ MB/s on a CMR HDD, 400+ MB/s on a SATA SSD,
# 1500+ MB/s on an NVMe drive. If you see <100 MB/s, something is wrong.

# 2. Read throughput — confirm no drvfs-style issues
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null  # Clear page cache
dd if=/mnt/cold/.test.bin of=/dev/null bs=1M status=progress
# Should show similar or higher numbers than write throughput.

# 3. Cleanup
rm /mnt/cold/.test.bin

# 4. Download a medium-sized model to exercise the full HF cache path
~/sentinel-factory/.venv/bin/python -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='Qwen/Qwen2.5-1.5B-Instruct',
    allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'],
)
print('Downloaded to:', path)
"
# Should complete in under a minute and land in /mnt/cold/huggingface-cache.

# 5. Load the model end-to-end to exercise the sustained-read path
~/sentinel-factory/.venv/bin/python -c "
from transformers import AutoModelForCausalLM
import torch, time
t0 = time.time()
m = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-1.5B-Instruct',
    torch_dtype=torch.float16,
    device_map='auto',
)
print(f'Loaded in {time.time()-t0:.1f}s')
"
# Should complete without hangs. If this hangs or takes more than a few
# seconds on a warm cache, something is wrong with the filesystem path.
```

All four steps should complete without hangs, without drvfs-style symptoms, without ever entering an uninterruptible `D` state. If any step hangs for more than 30 seconds without visible progress, you have a problem — stop and diagnose before running a real forge.

---

## Troubleshooting

### "mkfs.xfs: cannot operate / systemd is not running"

xfsprogs installed correctly; the warning is from a post-install hook that tries to register with systemd (which WSL2 does not run by default). The `mkfs.xfs` binary itself works fine regardless of the warning. Run `which mkfs.xfs && mkfs.xfs -V` to confirm the binary is present.

### "wsl --mount: this feature is not available in this version of Windows"

You need Windows 11 or Windows 10 version 2004+ with WSL 2.0 or newer. Check your version with `wsl --version` (note: this is a different command from `wsl --status`, which predates the feature). Upgrade WSL via the Microsoft Store if needed.

### The drive shows up in `lsblk` but `wipefs -a` says "permission denied"

Run with `sudo`. Most WSL2 distributions don't let non-root users write to raw block devices by default.

### The daemon hangs at ~70% of a model load even with xfs

Check `cat /proc/$PID/wchan` on the stuck thread. If it says `p9_client_rpc`, something is still going through drvfs — most likely the HF cache symlink is pointing to the old location. Run `ls -la ~/.cache/huggingface` and confirm it points at your xfs mount, not at `/mnt/c` or `/mnt/d`.

### The mount disappears after `wsl --shutdown`

Expected. WSL2 does not persist `wsl --mount --bare` calls across shutdowns. Re-run the mount from PowerShell after each shutdown, and re-run the `sudo mount -t xfs` step inside WSL. See Step 8 for a bootstrap script that automates the Linux side.

### I formatted the wrong drive

If you just formatted it, stop and do not write anything else to it. The NTFS partition table is gone but the file data may still be recoverable with tools like `testdisk` or `photorec`. Back up anything important from the correctly-formatted drive first, then use the recovery tools. This is why Step 1 of the setup is "confirm you have the right drive before proceeding" — there is no undo button for `mkfs.xfs`.

---

## Known lessons (from the BigMama 2026-04-10 incident)

Things we learned the hard way that are worth writing down so nobody else learns them the same way:

1. **drvfs hangs are silent and unkillable.** No error, no log, no SIGKILL. Only `wsl --shutdown` recovers. The only diagnostic is reading `/proc/$PID/wchan` for the stuck thread.
2. **The `p9_client_rpc` wchan value is the diagnostic fingerprint.** If you see it, drvfs is wedged and no amount of userspace intervention will help.
3. **Streaming-load is correct and necessary, but does not protect against drvfs hangs.** The fault is below the Python layer. Fix the filesystem, not the loader.
4. **`get_model_info`'s `fp16_gb` undercounts MoE models by a factor equal to the expert count.** Use on-disk safetensors sizes for streaming decisions, not computed estimates. (See sentinel-ai commit `3efd4b4`.)
5. **The heartbeat thread cannot save you from a kernel-level hang** because Python threads need the GIL and the GIL is held by the blocked C extension. Heartbeat hardening protects against blocking executor calls that *release* the GIL (the common case) but not against kernel-level hangs. Filesystem reliability is the only defense.
6. **xfs survived a power loss cleanly.** The filesystem journal replayed on mount, no corruption, no manual fsck required. This is what journaled filesystems are supposed to do and xfs does it right.

---

## See also

- `docs/HIVE-NODE-OPERATOR.md` — the broader operator playbook this doc is part of
- `docs/FACTORY-PROTOCOL.md` §0.1 (storage tiers) — the architectural contract this doc implements
- `docs/bootstrap-hive-node.sh` — the one-shot setup script that will eventually call into this doc's procedures automatically
- `continuum/docs/foreman/` (when it lands) — the Foreman role that will automate most of this
