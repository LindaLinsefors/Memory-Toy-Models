"""Run a single capacity-search sub-experiment as an isolated process.

This is the unit of parallelism: one (architecture, d, seed) combination. The
binary search inside ``find_max_facts`` is sequential, but many of these jobs
are fully independent, so the launcher fans a grid of them out across processes.

Each job writes its single result to its OWN .jsonl file (``<out-dir>/parts/``)
so that concurrent jobs never append to the same file at the same time. The
launcher merges the parts into per-series files afterwards.

Example (one job, on GPU):
    python run_one.py --series E4_CE_duplicate --d 64 --seed 42

Example (force CPU, e.g. to benchmark):
    python run_one.py --series E4_CE_duplicate --d 64 --device cpu
"""

import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # What to run
    p.add_argument("--series", required=True,
                   help="Logical series name (one line on the scaling plot), e.g. 'E4_CE_duplicate'.")
    p.add_argument("--d", type=int, required=True,
                   help="Model size: sets d_residual, d_ff, output_vocab_size=d, input_vocab_size=2d.")
    p.add_argument("--seed", type=int, default=42)

    # Architecture (defaults match E4: a pure embedding+head model)
    p.add_argument("--attention", action="store_true", default=False)
    p.add_argument("--ff", action="store_true", default=False)
    p.add_argument("--norms", action="store_true", default=False)

    # Where to write
    p.add_argument("--out-dir", default="E4")
    p.add_argument("--device", default=None,
                   help="torch device, e.g. 'cuda', 'cuda:0', 'cpu'. Defaults to cuda-if-available.")

    # Search / training hyperparameters (defaults match E4)
    p.add_argument("--loss-type", choices=["BCE", "CE"], default="CE")
    p.add_argument("--lr", type=float, nargs="+", default=[1e-2, 3e-3, 1e-3])
    p.add_argument("--n-epochs", type=int, default=50000)
    p.add_argument("--patience", type=int, default=5000)
    p.add_argument("--number-of-attempts", type=int, default=3)
    p.add_argument("--precision", type=int, default=None,
                   help="Binary-search precision. Default scales as 8*(d/16)^2 (matches E4).")

    # Logging
    p.add_argument("--no-wandb", action="store_true", default=False)
    return p.parse_args()


def main():
    # Force utf-8 so the ✓/✗ progress glyphs don't crash on a Windows cp1252 console.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    args = parse_args()

    # IMPORTANT: pin the device BEFORE importing models, because models.py sets
    # the global default device at import time.
    if args.device:
        os.environ["MTM_DEVICE"] = args.device

    # Imports deferred until after MTM_DEVICE is set.
    from models import ModelSettings
    from capacity_search import find_max_facts
    from log import log_result

    settings = ModelSettings(
        attention=args.attention,
        ff=args.ff,
        norms=args.norms,
        seed=args.seed,
        input_vocab_size=2 * args.d,
        output_vocab_size=args.d,
        d_residual=args.d,
        d_ff=args.d,
    )

    # Precision grows with capacity (~ d^2); matches the hand-picked E4 values
    # (d=16 -> 8, 32 -> 32, 64 -> 128, 128 -> 512).
    precision = args.precision if args.precision is not None else 8 * (args.d // 16) ** 2

    if args.loss_type == "CE":
        target_accuracy = "best_guess_accuracy"
        threshold_to_continue = 0.95
    else:
        target_accuracy = "accuracy"
        threshold_to_continue = 0.99

    log_to_wandb = not args.no_wandb
    tag = f"{args.series}_d{args.d}_s{args.seed}"
    wandb_group = f"{args.series}_d{args.d}"

    print(f"[run_one] series={args.series} d={args.d} seed={args.seed} "
          f"precision={precision} device={os.environ.get('MTM_DEVICE', 'auto')}")

    max_facts = find_max_facts(
        settings,
        precision=precision,
        n_epochs=args.n_epochs,
        lr=args.lr,
        patience=args.patience,
        number_of_attempts=args.number_of_attempts,
        log_to_wandb=log_to_wandb,
        wandb_log_every=10,
        wandb_group=wandb_group,
        verbose=True,
        target_accuracy=target_accuracy,
        threshold_to_continue=threshold_to_continue,
        loss_type=args.loss_type,
    )

    # Write to a per-job file so parallel jobs never contend on the same file.
    parts_dir = os.path.join(args.out_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    log_result(args.series, max_facts, settings, os.path.join(parts_dir, tag), extra={
        "series": args.series,
        "seed": args.seed,
        "n_epochs": args.n_epochs,
        "lr": args.lr,
        "patience": args.patience,
        "target_accuracy": target_accuracy,
        "threshold_to_continue": threshold_to_continue,
        "precision": precision,
        "number_of_attempts": args.number_of_attempts,
        "loss_type": args.loss_type,
        "wandb_group": wandb_group,
    })

    print(f"[run_one] DONE series={args.series} d={args.d} seed={args.seed} max_facts={max_facts}")


if __name__ == "__main__":
    main()
