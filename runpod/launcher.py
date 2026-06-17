"""Fan a grid of capacity-search jobs out across parallel worker processes.

Each job is an independent ``run_one.py`` invocation (one architecture, d, seed).
Because the models are tiny, you can run many concurrently without meaningfully
slowing each other down. Jobs write to their own files; this script merges them
into per-series ``.jsonl`` files that E4.py's plotting code reads.

Designed for "start it, walk away, read results tomorrow":
  - Run it inside tmux so it survives SSH disconnects.
  - Per-job stdout/stderr goes to <out-dir>/logs/<tag>.log (tail any single job).
  - Metrics also stream to Weights & Biases as usual.

Examples
--------
Whole default E4 grid on the GPU, 8 jobs at a time:
    python launcher.py --n-parallel 8

Same grid on CPU cores (benchmark), one job per ~4 cores:
    python launcher.py --device cpu --n-parallel 8

Wider sweep with repeats for error bars:
    python launcher.py --d 16 32 64 128 256 --seeds 0 1 2 --n-parallel 16
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--series", default=None,
                   help="Series base name. Default: 'E4_<loss>[_duplicate]'.")
    p.add_argument("--d", type=int, nargs="+", default=[16, 32, 64, 128],
                   help="Model sizes to sweep.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="Seeds per model size (multiple => repeats for error bars).")
    p.add_argument("--loss-type", choices=["BCE", "CE"], default="CE")

    p.add_argument("--n-parallel", type=int, default=4,
                   help="Max concurrent jobs.")
    p.add_argument("--device", default=None,
                   help="torch device for every job ('cuda', 'cpu', ...). Default: cuda-if-available.")
    p.add_argument("--threads-per-job", type=int, default=None,
                   help="CPU threads per job. Default: cpu_count//n_parallel for --device cpu, else 1.")

    p.add_argument("--out-dir", default="E4")
    p.add_argument("--no-wandb", action="store_true", default=False)

    # Pass-through training knobs (forwarded to every job).
    p.add_argument("--n-epochs", type=int, default=50000)
    p.add_argument("--patience", type=int, default=5000)
    p.add_argument("--number-of-attempts", type=int, default=3)
    p.add_argument("--lr", type=float, nargs="+", default=[1e-2, 3e-3, 1e-3])

    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Print the jobs that would run, then exit.")
    return p.parse_args()


def build_series_name(args) -> str:
    if args.series:
        return args.series
    name = f"E4_{args.loss_type}"
    if len(args.seeds) > 1:
        name += "_multiseed"
    return name


def job_env(args, n_parallel: int) -> dict:
    """Environment for each child: pin device and cap threads to avoid oversubscription."""
    env = os.environ.copy()
    if args.device:
        env["MTM_DEVICE"] = args.device

    if args.threads_per_job is not None:
        threads = args.threads_per_job
    elif args.device == "cpu":
        threads = max(1, (os.cpu_count() or 1) // max(1, n_parallel))
    else:
        threads = 1
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[var] = str(threads)
    return env


def merge_parts(out_dir: str, series: str):
    """Concatenate per-job parts for a series into <out-dir>/<series>.jsonl.

    Records are sorted by (d_residual, seed) so the scaling plot is monotonic in d.
    """
    parts_dir = os.path.join(out_dir, "parts")
    if not os.path.isdir(parts_dir):
        return
    records = []
    for fname in os.listdir(parts_dir):
        if fname.startswith(series + "_d") and fname.endswith(".jsonl"):
            with open(os.path.join(parts_dir, fname), "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
    if not records:
        return
    records.sort(key=lambda r: (r["settings"]["d_residual"], r["settings"]["seed"]))
    merged = os.path.join(out_dir, f"{series}.jsonl")
    with open(merged, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[launcher] merged {len(records)} record(s) -> {merged}")


def main():
    args = parse_args()
    series = build_series_name(args)
    os.makedirs(os.path.join(args.out_dir, "logs"), exist_ok=True)

    grid = list(itertools.product(args.d, args.seeds))
    jobs = []
    for d, seed in grid:
        tag = f"{series}_d{d}_s{seed}"
        cmd = [
            sys.executable, "-u", "run_one.py",
            "--series", series,
            "--d", str(d),
            "--seed", str(seed),
            "--out-dir", args.out_dir,
            "--loss-type", args.loss_type,
            "--n-epochs", str(args.n_epochs),
            "--patience", str(args.patience),
            "--number-of-attempts", str(args.number_of_attempts),
            "--lr", *[str(x) for x in args.lr],
        ]
        if args.device:
            cmd += ["--device", args.device]
        if args.no_wandb:
            cmd += ["--no-wandb"]
        jobs.append((tag, cmd))

    print(f"[launcher] series={series} jobs={len(jobs)} n_parallel={args.n_parallel} "
          f"device={args.device or 'cuda-if-available'}")
    for tag, cmd in jobs:
        print(f"  - {tag}: {' '.join(cmd)}")
    if args.dry_run:
        return

    env = job_env(args, args.n_parallel)
    running = {}  # tag -> (Popen, log_file_handle)
    pending = list(jobs)
    failures = []
    started = time.time()

    def launch(tag, cmd):
        log_path = os.path.join(args.out_dir, "logs", f"{tag}.log")
        fh = open(log_path, "w", encoding="utf-8")
        fh.write(f"# {tag}\n# started {datetime.now().isoformat()}\n# {' '.join(cmd)}\n\n")
        fh.flush()
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        running[tag] = (proc, fh)
        print(f"[launcher] START {tag} (pid {proc.pid}) -> {log_path}")

    while pending or running:
        while pending and len(running) < args.n_parallel:
            tag, cmd = pending.pop(0)
            launch(tag, cmd)

        time.sleep(1.0)

        for tag in list(running):
            proc, fh = running[tag]
            ret = proc.poll()
            if ret is not None:
                fh.close()
                del running[tag]
                status = "OK" if ret == 0 else f"FAILED (exit {ret})"
                if ret != 0:
                    failures.append(tag)
                done = len(jobs) - len(pending) - len(running)
                print(f"[launcher] {status} {tag}  [{done}/{len(jobs)} done]")

    merge_parts(args.out_dir, series)

    elapsed = time.time() - started
    print(f"\n[launcher] all done in {elapsed/60:.1f} min. "
          f"{len(jobs) - len(failures)}/{len(jobs)} succeeded.")
    if failures:
        print("[launcher] failures (see their logs): " + ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
