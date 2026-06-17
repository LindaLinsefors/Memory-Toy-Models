# Cloud compute setup — Memory Toy Models

Target: **RunPod Secure Cloud**, an **A100 80GB** pod with a **persistent network
volume**, on-demand (not spot). Goal: start a sweep today, read results tomorrow,
run many tiny jobs in parallel.

The repo is already prepped for this:
- `device.py` — picks `cuda`/`cpu` automatically; override with `MTM_DEVICE`.
- `run_one.py` — one isolated capacity-search job (one architecture, `d`, seed).
- `launcher.py` — fans a grid of jobs across parallel worker processes, then
  merges results into per-series `.jsonl` files for the plots in `E4.py`.
- `requirements.txt` — minimal deps (torch comes from the pod image).

---

## 1. Account, billing, SSH key (one-time)

1. Sign up at <https://www.runpod.io> and verify your email.
2. **Billing → add credit.** RunPod is prepaid; add e.g. $25–50 to start. An A100
   80GB on Secure Cloud is roughly **$1.6–2.0/hr** while running; a stopped pod
   only pays for storage.
3. **Settings → SSH Public Keys.** Paste your public key so you get real SSH
   (needed for VS Code Remote-SSH). If you don't have one yet, on your laptop:
   ```powershell
   ssh-keygen -t ed25519 -C "linda.linsefors@gmail.com"
   # then paste the contents of  C:\Users\linda\.ssh\id_ed25519.pub  into RunPod
   ```
4. *(Optional, for scripting)* **Settings → API Keys** → create one if you later
   want to launch pods with `runpodctl` instead of the web UI.

## 2. Create the persistent network volume (one-time)

This is what survives across days and pod restarts — your code and `.jsonl`
results live here.

1. **Storage → Network Volumes → New Network Volume.**
2. Pick a **datacenter that has A100 80GB** (e.g. a `US` or `EU` region that lists
   A100 availability). **Write the region down** — your pod must be in the *same*
   region as the volume.
3. Size: **50 GB** is plenty (results are tiny; this is mostly for the conda/pip
   env and the repo). Storage is ~$0.05–0.07/GB/month.
4. Name it e.g. `mtm-data`.

## 3. Deploy the A100 pod

1. **Pods → Deploy → Secure Cloud.**
2. Filter to your volume's region, select **A100 80GB (PCIe or SXM)**.
3. **Template:** choose a **RunPod PyTorch 2.x** image (ships CUDA + torch, so you
   don't reinstall torch).
4. **Attach Network Volume:** select `mtm-data` (mounts at `/workspace`).
5. **Instance Pricing:** **On-Demand** (not Spot/Interruptible) — an overnight
   sweep must not be evicted.
6. Deploy. When it's "Running", open **Connect** to get the SSH command.

## 4. Connect

You wanted both interactive `#%%` cells *and* headless `python` runs — here's both.

**a) VS Code Remote-SSH (for interactive `#%%` work)**

Add the pod to `C:\Users\linda\.ssh\config` (copy host/port from the Connect panel):
```
Host mtm-pod
    HostName <pod-ip-from-runpod>
    User root
    Port <port-from-runpod>
    IdentityFile C:\Users\linda\.ssh\id_ed25519
```
Then in VS Code: **Remote-SSH: Connect to Host → mtm-pod**, open `/workspace`,
install the Python + Jupyter extensions in the remote, and run the `#%%` cells in
`E4.py` exactly like you do locally. Pick the pod's Python interpreter.

**b) Plain terminal (for headless sweeps)**

Just `ssh mtm-pod` from a local terminal.

## 5. One-time setup on the pod

```bash
cd /workspace

# Get the code. Easiest: push this repo to GitHub, then:
git clone https://github.com/<you>/memory-toy-models.git
cd memory-toy-models
# (Alternatively rsync from your laptop — see "Syncing code" below.)

pip install -r requirements.txt   # torch already in the image
wandb login                       # paste key from https://wandb.ai/authorize

# Quick sanity check that GPU + the pipeline work (no W&B, tiny run):
python run_one.py --series check --d 16 --no-wandb \
    --n-epochs 50 --patience 10 --number-of-attempts 1 --out-dir _check
rm -rf _check
```

## 6. Run experiments

**Interactive:** open `E4.py` in Remote-SSH and run the cells.

**Headless, parallel, the whole E4 grid on the GPU (8 jobs at once):**
```bash
tmux new -s sweep        # so it survives disconnects
python launcher.py --n-parallel 8
#  Ctrl-b then d  to detach. Log out, close your laptop.
```

Reattach later (next day) with `tmux attach -t sweep`. Per-job progress is in
`E4/logs/<tag>.log`; results merge into `E4/<series>.jsonl`; metrics are in W&B.

**Wider sweep with repeats (error bars):**
```bash
python launcher.py --d 16 32 64 128 256 --seeds 0 1 2 --n-parallel 16
```

## 7. "Start today, read tomorrow"

- The sweep runs inside `tmux` on the pod, so it keeps going after you disconnect.
- **Do not `stop` the pod while a job is running** — stopping kills the processes.
  Leave it running overnight (that's the GPU-hour cost you're fine with).
- Next day: `tmux attach`, or just read `E4/<series>.jsonl` / the W&B project
  "Memory Toy Models".
- Pull results back to your laptop:
  ```powershell
  scp -P <port> -r root@<pod-ip>:/workspace/memory-toy-models/E4/*.jsonl `
      "c:\Users\linda\VS Code Workspace\Memory Toy Models\E4\"
  ```
  Then run the plotting cell at the bottom of `E4.py` locally.

## 8. Cost control

- **Stop the pod when idle:** GPU billing stops; the network volume (and its
  contents) persist. Restart it next session and everything is where you left it.
- **Terminate** only if you're done for a while — the *volume* still keeps your
  data even after the pod is terminated.

## 9. GPU vs CPU — quick benchmark before committing a strategy

The models are tiny, so CPU-core parallelism sometimes beats piling jobs on one
GPU. The A100 box has many vCPUs, so test both and use whichever is faster:
```bash
# GPU: many jobs share the one A100
time python launcher.py --series bench_gpu --d 16 32 64 --device cuda \
    --n-parallel 6 --n-epochs 2000 --number-of-attempts 1

# CPU: one job per few cores
time python launcher.py --series bench_cpu --d 16 32 64 --device cpu \
    --n-parallel 12 --n-epochs 2000 --number-of-attempts 1
```
Use the faster one for the real run. As a rule of thumb, push `--n-parallel` up
until wall-clock per job stops improving.

---

### Syncing code without GitHub (optional)

From your laptop (requires `rsync`, e.g. via Git Bash/WSL):
```bash
rsync -avz -e "ssh -p <port>" --exclude wandb/ --exclude .git/ \
    "/c/Users/linda/VS Code Workspace/Memory Toy Models/" \
    root@<pod-ip>:/workspace/memory-toy-models/
```
