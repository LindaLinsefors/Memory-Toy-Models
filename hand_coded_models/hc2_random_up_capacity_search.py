"""Capacity search for RandomUpModel2 (RANDOM frozen up matrix + trained down), on Modal (GPU).

Mirror of hc2_hybrid_capacity_search.py with the hand-coded up matrix replaced
by a random one (see hc2_random_up.py): the up matrix is drawn once per attempt
(nn.Linear-style uniform init, frozen) and the down matrix + bias are trained
with full-batch CE loss — the same recipe as the hybrid search (plain Adam,
lr=1e-2, up to 5000 epochs, early stopping; see hc2_hybrid.train_down_matrix).
A cell's score is the best accuracy observed during training.

Because nothing about the up matrix is hand-coded, there is no S, top_n, or
top_fraction to sweep — and no connection matrices to build. A "grid" for one
n_facts is therefore just the n_attempts training runs, each with a different
init_seed (drawing a fresh random up matrix and down init). Everything else
mirrors the hybrid search: the same 11 attempts, the same any/all/most
reduction over them, the same (d, accuracy_threshold, any_all_most) configs,
and the same binary search over n_facts with a 2%-relative stop
(hi - lo < 0.02 * hi).

Outputs go to their own locations so they never mix with other runs: grids in
hc2_sweep_results/randomup_grids/ and the capacity log in
capacity_search_results_randomup.json.

Run it with:
    python -m modal run hand_coded_models/hc2_random_up_capacity_search.py
"""

import os
import json
import glob
from collections import defaultdict
from datetime import datetime

import numpy as np
import modal


# Fixed seed for generate_facts (the facts never change), and this file's dir.
seed = 42
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "hc2_sweep_results")

# Grid files live in their own subfolder; only the capacity_search_results_*
# logs sit directly in RESULTS_DIR.
GRIDS_DIR = os.path.join(RESULTS_DIR, "randomup_grids")


# ── Settings ──────────────────────────────────────────────────────────────────
n_attempts = 11

# Training recipe for the down matrix (see hc2_hybrid.train_down_matrix).
N_EPOCHS = 5000
LR = 1e-2
PATIENCE = 100

# Where the training containers run. Same pattern as capacity_search.py: the
# base function is CPU-only and a GPU is added at call time via .with_options
# (a GPU can be added that way but not unset, so the base has no gpu=).
# models.py sets the default device to cuda-if-available inside the container.
device = "gpu"     # "gpu" or "cpu"
MODAL_GPU = "T4"   # GPU type used when device == "gpu"

# Each capacity search runs one (d, accuracy_threshold, any_all_most) config.
CONFIGS = [
    dict(d=d, accuracy_threshold=thr, any_all_most=aam)
    for d in [16, 32, 64, 128, 256]
    for thr in [0.9, 1.0]
    for aam in ["any", "all", "most"]
]

testing = False  # small/cheap end-to-end validation run

# Binary search stops when the bracket is within this fraction of its top end
# (hi - lo < PRECISION_FRACTION * hi), giving ~2% relative resolution on max_facts.
PRECISION_FRACTION = 0.02


def _init_seed(attempt):
    """Per-attempt seed for the random up matrix and the down init — the only
    source of variation between attempts (the facts are fixed)."""
    return attempt


# ── Modal app + image ─────────────────────────────────────────────────────────
app = modal.App("hc2-random-up-capacity-search")

# Self-contained image: ships hc2_random_up.py + hc2_hybrid.py (for
# train_down_matrix; it imports hc2.py) + the repo-root modules they import.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "wandb")
    .add_local_python_source("models", "device")
    .add_local_file(os.path.join(_HERE, "hc2.py"), "/root/hc2.py")
    .add_local_file(os.path.join(_HERE, "hc2_hybrid.py"), "/root/hc2_hybrid.py")
    .add_local_file(os.path.join(_HERE, "hc2_random_up.py"), "/root/hc2_random_up.py")
)


@app.function(image=image, timeout=86400)
def _run_one(cell):
    """Train + evaluate one attempt for a given n_facts.

    Builds a RandomUpModel2 (fresh random frozen up matrix per attempt) and
    trains the down matrix with CE loss; returns the best accuracy
    seen during training."""
    import sys
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    from hc2_random_up import RandomUpModel2
    from hc2_hybrid import train_down_matrix

    model = RandomUpModel2(
        input_vocab_size=cell["input_vocab_size"],
        output_vocab_size=cell["output_vocab_size"],
        n_facts=cell["n_facts"],
        d_ff=cell["d_ff"],
        seed=cell["seed"],
        init_seed=cell["init_seed"],
    )
    accuracy, epochs_run = train_down_matrix(
        model, n_epochs=N_EPOCHS, lr=LR, patience=PATIENCE)
    return {
        "n_facts": cell["n_facts"],
        "attempt": cell["attempt"],
        "init_seed": cell["init_seed"],
        "accuracy": accuracy,
        "epochs_run": epochs_run,
    }


# ── Result caching (grid files: hc2_randomup_d{d}_nfacts{nf}.json) ─────────────

def _grid_glob():
    return os.path.join(GRIDS_DIR, "hc2_randomup_d*_nfacts*.json")


def _load_cached_records(d, n_facts):
    """Pool every grid record for (d, n_facts) from disk."""
    records = []
    for path in sorted(glob.glob(_grid_glob())):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        s = payload.get("settings", {})
        if s.get("d") == d and s.get("n_facts") == n_facts:
            records.extend(payload["results"])
    return records


def _covers(records, n_attempts):
    """True if `records` already hold >= n_attempts runs for this n_facts."""
    return len(records) >= n_attempts


def _build_cells_for_n_facts(d, n_facts, n_attempts):
    """One Modal task per attempt (there is no S / knob dimension)."""
    cells = []
    for a in range(n_attempts):
        cells.append({
            "d": d,
            "attempt": a,
            "n_facts": n_facts,
            "init_seed": _init_seed(a),
            "input_vocab_size": 2 * d,
            "output_vocab_size": d,
            "d_ff": d,
            "seed": seed,
        })
    return cells


def _save_records(d, n_facts, records, n_attempts):
    """Write a grid result file (one per n_facts) into the grid subfolder."""
    os.makedirs(GRIDS_DIR, exist_ok=True)
    payload = {
        "settings": {
            "d": d,
            "n_facts": n_facts,
            "n_attempts": n_attempts,
            "input_vocab_size": 2 * d,
            "output_vocab_size": d,
            "d_ff": d,
            "seed": seed,
            "metric": "accuracy",
            "model": "RandomUpModel2",
            "n_epochs": N_EPOCHS,
            "lr": LR,
            "patience": PATIENCE,
        },
        "results": records,
    }
    out_path = os.path.join(GRIDS_DIR, f"hc2_randomup_d{d}_nfacts{n_facts}.json")
    base, ext = os.path.splitext(out_path)
    i = 1
    while os.path.exists(out_path):
        out_path = f"{base}_({i}){ext}"
        i += 1
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


# ── Grid evaluation ───────────────────────────────────────────────────────────

def _evaluate_n_facts(d, n_attempts, n_facts,
                      accuracy_threshold=1.0, any_all_most="any", verbose=False):
    """Decide whether n_facts is storable.

    With no hyper-parameter grid, the decision is simply the any/all/most
    reduction over the n_attempts training runs. Returns (success, score).
    Uses the disk cache if a covering grid already exists; otherwise runs the
    training cells on Modal and saves.
    """
    if any_all_most not in ("any", "all", "most"):
        raise ValueError("any_all_most must be 'any', 'all', or 'most'")

    records = _load_cached_records(d, n_facts)
    if _covers(records, n_attempts):
        if verbose:
            print(f"    n_facts={n_facts}: loaded grid from cache ({len(records)} records)")
    else:
        cells = _build_cells_for_n_facts(d, n_facts, n_attempts)
        if verbose:
            print(f"    n_facts={n_facts}: running {len(cells)} training cells on Modal ({device}) ...")
        if device == "gpu":
            fn = _run_one.with_options(gpu=MODAL_GPU)
        elif device == "cpu":
            fn = _run_one
        else:
            raise ValueError("device must be 'cpu' or 'gpu'")
        records = list(fn.map(cells))
        path = _save_records(d, n_facts, records, n_attempts)
        if verbose:
            print(f"    n_facts={n_facts}: wrote {len(records)} records to {path}")

    accs = [r.get("accuracy", r.get("best_guess_accuracy")) for r in records]

    # Reduce the runs to one score matching the rule, so success <=> score >= threshold.
    #   any -> max over runs ; all -> min over runs ; most -> median over runs
    reduce_fn = {"any": np.max, "all": np.min, "most": np.median}[any_all_most]
    score = float(reduce_fn(accs))

    success = score >= accuracy_threshold
    if verbose:
        print(f"    n_facts={n_facts}: {any_all_most} score={score:.4f} "
              f"{'>=' if success else '<'} {accuracy_threshold} "
              f"-> {'PASS' if success else 'fail'}")
    return success, score


# ── Binary search over n_facts ────────────────────────────────────────────────

def find_max_facts(d, n_attempts,
                   accuracy_threshold=1.0, any_all_most="any", verbose=False):
    """Binary-search the maximum storable n_facts. max_possible = (2*d)^2 = 4*d**2.

    Stops when the bracket is within PRECISION_FRACTION of its top end
    (hi - lo < PRECISION_FRACTION * hi). Returns (best_n_facts, best_score) with
    best_score the any/all/most score at the largest passing n_facts.
    """
    max_possible = 4 * d ** 2
    lo, hi = 1, max_possible
    best = 0
    best_score = None

    if verbose:
        print(f"Searching for max storable facts in [{lo}, {hi}]  (d={d})\n")

    while hi - lo >= PRECISION_FRACTION * hi:
        mid = (lo + hi) // 2
        if verbose:
            print(f"Trying n_facts = {mid} ...")
        success, score = _evaluate_n_facts(
            d, n_attempts, mid,
            accuracy_threshold=accuracy_threshold,
            any_all_most=any_all_most, verbose=verbose,
        )
        if success:
            best, best_score = mid, score
            lo = mid + 1
            if verbose:
                print(f"OK  stored {mid} facts. Now searching: {lo} - {hi}\n")
        else:
            hi = mid - 1
            if verbose:
                print(f"X   failed at {mid} facts. Now searching: {lo} - {hi}\n")

    if hi == max_possible:  # loop never tests the top end
        if verbose:
            print(f"Trying n_facts = {max_possible} (max possible) ...")
        success, score = _evaluate_n_facts(
            d, n_attempts, max_possible,
            accuracy_threshold=accuracy_threshold,
            any_all_most=any_all_most, verbose=verbose,
        )
        if success:
            best, best_score = max_possible, score

    if verbose:
        print(f"\nMax storable facts: {best}  (score = {best_score})")
    return best, best_score


# ── Result logging + driver ───────────────────────────────────────────────────
CAPACITY_RESULTS_PATH = os.path.join(RESULTS_DIR, "capacity_search_results_randomup.json")


def _append_capacity_result(d, max_facts, best_score, accuracy_threshold,
                            any_all_most, n_attempts):
    """Append one JSON line summarising a run; never touches previous lines."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "d": d,
        "max_facts": max_facts,
        "best_score": best_score,
        "accuracy_threshold": accuracy_threshold,
        "any_all_most": any_all_most,
        "n_attempts": n_attempts,
        "precision_fraction": PRECISION_FRACTION,
        "model": "RandomUpModel2",
        "n_epochs": N_EPOCHS,
        "lr": LR,
        "patience": PATIENCE,
    }
    with open(CAPACITY_RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return CAPACITY_RESULTS_PATH


@app.local_entrypoint()
def main():
    configs = CONFIGS
    attempts = n_attempts
    if testing:
        # Cheap end-to-end validation: smallest d, few attempts.
        configs = [dict(d=16, accuracy_threshold=1.0, any_all_most="any")]
        attempts = 2

    print(f"Running {len(configs)} capacity search config(s).")
    for cfg in configs:
        d = cfg["d"]
        thr = cfg["accuracy_threshold"]
        aam = cfg["any_all_most"]

        print(f"\n===== d={d}, accuracy_threshold={thr}, any_all_most={aam} =====")
        best, score = find_max_facts(
            d, attempts,
            accuracy_threshold=thr, any_all_most=aam, verbose=False,
        )
        print(f"d={d}: max_facts={best}, score={score}")
        path = _append_capacity_result(
            d, best, score, thr, aam, attempts)
        print(f"Appended result to {path}")
