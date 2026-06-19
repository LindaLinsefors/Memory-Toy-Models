"""Capacity search for HandCodedModel2.

Binary-search the largest n_facts that HandCodedModel2 can store, where "can
store" is decided by sweeping (top_fraction, S) and asking whether ANY cell of
that grid reaches `accuracy_threshold` (under the any/all/most rule).

The expensive grid evaluation reuses the Modal fan-out from hc2_sweep.py: each
(S, top_fraction, attempt) point becomes one container that builds its own fresh
connection matrix. Grids are cached to hand_coded_models/hc2_sweep_results/ (the
same files hc2_sweep.py / hc2_sweep_plot.py use), so a value of n_facts already
swept on disk is loaded instead of recomputed.

Run it with:
    python -m modal run hand_coded_models/hc2_capacity_search.py
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


# Settings used when running this file.
d = 32
accuracy_threshold = 0.9
any_all_most = "all"


n_attempts = 11
precision = 8 if d == 16 else 8 * 4 if d == 32 else 8 * 4 * 4 if d == 64 else 8 * 4 * 4 * 4

if d == 16:
    S_sweep = [1,2,3,4,5,6,7,8,9,10]
elif d == 32:
    S_sweep = [1,2,3,4,5,6,7,8,9,10,11,13]
elif d == 64:
    S_sweep = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
elif d == 128:
    S_sweep = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

top_fraction_sweep = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


app = modal.App("hc2-capacity-search")

# Self-contained container image (mirrors hc2_sweep.py): ships hc2.py plus the
# repo-root modules it imports, so `import hc2` works inside the container. We do
# NOT import anything from hc2_sweep at module level — Modal re-imports THIS file
# inside every container, and `hand_coded_models` is not on the container path.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "wandb")
    .add_local_python_source("models", "device")
    .add_local_file(os.path.join(_HERE, "hc2.py"), "/root/hc2.py")
)


@app.function(image=image, timeout=86400)
def _run_one(args: dict) -> dict:
    """Evaluate ONE model: a single (n_facts, S, top_fraction, attempt) point.

    Self-contained copy of hc2_sweep._run_one so this file is an independent Modal
    app. Builds a fresh connection matrix from a unique conn_seed; the facts are
    fixed by `seed`, so only the matrix (and tie-breaking) varies across attempts.
    """
    import sys
    import warnings
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    import torch
    from hc2 import (
        HandCodedModel2,
        HandCodedModel2Settings,
        make_connection_matrix,
    )

    S = args["S"]
    conn_seed = args["conn_seed"]
    torch.manual_seed(conn_seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conn = make_connection_matrix(D=args["d_ff"], T=args["output_vocab_size"],
                                      S=S, seed=conn_seed)

    settings = HandCodedModel2Settings(
        input_vocab_size=args["input_vocab_size"],
        output_vocab_size=args["output_vocab_size"],
        n_facts=args["n_facts"],
        seed=args["seed"],
        d_ff=args["d_ff"],
        n_neurons_per_label=S,
        use_top_no_top_fraction="top_fraction",
        top_fraction=args["top_fraction"],
    )
    model = HandCodedModel2(settings, precomputed_conn=conn)
    _, best_guess_accuracy, _, _ = model.evaluate()
    return {
        "n_facts": args["n_facts"],
        "S": S,
        "top_fraction": args["top_fraction"],
        "attempt": args["attempt"],
        "conn_seed": conn_seed,
        "best_guess_accuracy": best_guess_accuracy,
    }


RESULTS_DIR = os.path.join(_HERE, "hc2_sweep_results")


# ── Result caching ────────────────────────────────────────────────────────────

def _load_cached_records(d, n_facts):
    """Pool every record for (d, n_facts) from the result files on disk.

    Mirrors hc2_sweep_plot._load_records: scans RESULTS_DIR, keeps files whose
    settings match (d, n_facts), and concatenates their `results` lists. Skips
    test_*.json. Returns a (possibly empty) list of record dicts.
    """
    records = []
    # Match only grid files by their naming pattern (hc2_sweep_d{d}_nfacts{nf}.json
    # and its _({i}) variants). This ignores any other .json in the folder — e.g.
    # the JSONL capacity-search log or test_* runs — so co-location is harmless.
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "hc2_sweep_d*_nfacts*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        s = payload.get("settings", {})
        if s.get("d") == d and s.get("n_facts") == n_facts:
            records.extend(payload["results"])
    return records


def _covers(records, S_sweep, top_fraction_sweep, n_attempts):
    """True if `records` already hold >= n_attempts runs for every (S, tf) wanted."""
    counts = defaultdict(int)
    for r in records:
        counts[(r["S"], r["top_fraction"])] += 1
    for S in S_sweep:
        for tf in top_fraction_sweep:
            if counts[(S, tf)] < n_attempts:
                return False
    return True


def _build_cells_for_n_facts(d, n_facts, n_attempts, top_fraction_sweep, S_sweep,
                             seed_offset=0):
    """Build the per-evaluation work units for one n_facts (see hc2_sweep._build_cells).

    One cell per (S, top_fraction, attempt); each gets a unique conn_seed so its
    connection matrix is generated independently.
    """
    cells = []
    idx = seed_offset
    for S in S_sweep:
        for top_fraction in top_fraction_sweep:
            for attempt in range(n_attempts):
                cells.append({
                    "n_facts": n_facts,
                    "S": S,
                    "top_fraction": top_fraction,
                    "attempt": attempt,
                    "conn_seed": seed + 1 + idx,
                    "d_ff": d,
                    "seed": seed,
                    "input_vocab_size": 2 * d,
                    "output_vocab_size": d,
                })
                idx += 1
    return cells


def _save_records(d, n_facts, records, n_attempts, top_fraction_sweep, S_sweep):
    """Write a grid result file in the schema hc2_sweep.py uses (one per n_facts)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "settings": {
            "d": d,
            "n_facts": n_facts,
            "n_attempts": n_attempts,
            "top_fraction_sweep": top_fraction_sweep,
            "S_sweep": S_sweep,
            "input_vocab_size": 2 * d,
            "output_vocab_size": d,
            "d_ff": d,
            "seed": seed,
            "metric": "best_guess_accuracy",
        },
        "results": records,
    }
    out_path = os.path.join(RESULTS_DIR, f"hc2_sweep_d{d}_nfacts{n_facts}.json")
    base, ext = os.path.splitext(out_path)
    i = 1
    while os.path.exists(out_path):
        out_path = f"{base}_({i}){ext}"
        i += 1
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


# ── Grid evaluation ───────────────────────────────────────────────────────────

def _evaluate_n_facts(
        d,
        n_attempts,
        n_facts,
        top_fraction_sweep,
        S_sweep,
        accuracy_threshold = 1.0,
        any_all_most = "any",
        verbose = True,
):
    '''
    Evaluate the model's performance for a given number of facts.

    Args:
        d (int): The dimensionality of the model.
        n_attempts (int): The number of attempts to evaluate.
        n_facts (int): The number of facts to evaluate.
        top_fraction_sweep (list): List of top fraction values to sweep.
        S_sweep (list): List of S values to sweep.
        accuracy_threshold (float, optional): Accuracy threshold for evaluation. Defaults to 1.0.
        any_all_most (str, optional): Evaluation mode ('any', 'all', 'most'). Defaults to "any".

    Returns:
        success (bool): Whether the evaluation was successful based on the accuracy threshold.
        best_combination (tuple): The (top_fraction, S) combination that had the best accuracy.

    Method:
    1) Check if there is a file in hc2_sweep_results with the results
    for this (d, n_attempts, n_facts, top_fraction_sweep, S_sweep)
    combination. If so, load it, if not generate it, using functions from
    hc2_sweep.py. Use modal.
    2) Find if there any (top_fraction, S) combinations that have
    accuracy >= accuracy_threshold.
    2a) For any_all_most = "all", success=True if all runs for any
    (top_fraction, S) combination have accuracy >= accuracy_threshold.
    2b) For any_all_most = "any", success=True if any run for any
    (top_fraction, S) combination has accuracy >= accuracy_threshold.
    2c) For any_all_most = "most", success=True if most runs for any
    (top_fraction, S) combination has accuracy >= accuracy_threshold.
    3) Return success, and the (top_fraction, S) combination that had
    the best accuracy.
    '''
    if any_all_most not in ("any", "all", "most"):
        raise ValueError("any_all_most must be 'any', 'all', or 'most'")

    # 1) Use cached results if the grid is already on disk; otherwise run it on Modal.
    records = _load_cached_records(d, n_facts)
    if _covers(records, S_sweep, top_fraction_sweep, n_attempts):
        if verbose:
            print(f"    n_facts={n_facts}: loaded grid from cache "
                  f"({len(records)} records)")
    else:
        cells = _build_cells_for_n_facts(
            d, n_facts, n_attempts, top_fraction_sweep, S_sweep)
        if verbose:
            print(f"    n_facts={n_facts}: running {len(cells)} evaluations on Modal ...")
        records = list(_run_one.map(cells))
        path = _save_records(d, n_facts, records, n_attempts,
                             top_fraction_sweep, S_sweep)
        if verbose:
            print(f"    n_facts={n_facts}: wrote {len(records)} records to {path}")

    # 2) Collect each (S, top_fraction) cell's runs, restricted to the wanted sweep.
    wanted_S = set(S_sweep)
    wanted_tf = set(top_fraction_sweep)
    cells = defaultdict(list)
    for r in records:
        if r["S"] in wanted_S and r["top_fraction"] in wanted_tf:
            cells[(r["S"], r["top_fraction"])].append(r["best_guess_accuracy"])

    # Reduce each cell to one "score" matching the any/all/most rule, so that
    #   success  <=>  best cell score >= accuracy_threshold.
    #   any  -> a single run is enough        -> max  over runs
    #   all  -> every run must pass           -> min  over runs
    #   most -> a majority must pass          -> median over runs
    if any_all_most == "any":
        reduce_fn = np.max
    elif any_all_most == "all":
        reduce_fn = np.min
    else:
        reduce_fn = np.median

    best_score = -1.0
    best_combination = None
    for (S, tf), accs in cells.items():
        score = float(reduce_fn(accs))
        if score > best_score:
            best_score = score
            best_combination = (tf, S)

    # 3) Success iff the best cell's score clears the threshold.
    success = best_score >= accuracy_threshold
    if verbose:
        tf, S = best_combination
        print(f"    n_facts={n_facts}: best {any_all_most} cell "
              f"(top_fraction={tf}, S={S}) score={best_score:.4f} "
              f"{'>=' if success else '<'} {accuracy_threshold}  "
              f"-> {'PASS' if success else 'fail'}")

    return success, best_combination


# ── Binary search over n_facts ────────────────────────────────────────────────

def find_max_facts(
        d,
        n_attempts,
        top_fraction_sweep,
        S_sweep,
        precision=1,
        accuracy_threshold=1.0,
        any_all_most="any",
        verbose=True,
):
    """Binary-search the maximum n_facts whose (top_fraction, S) grid passes.

    Inspired by find_max_facts in capacity_search.py. max_possible is the number
    of distinct two-token inputs available: (2*d)^2 = 4*d**2.

    Returns (best_n_facts, best_combination) where best_combination is the
    (top_fraction, S) that scored best at best_n_facts (or None if nothing passed).
    """
    max_possible = 4 * d ** 2
    lo, hi = 1, max_possible
    best = 0
    best_combo = None

    if verbose:
        print(f"Searching for max storable facts in [{lo}, {hi}]  (d={d})\n")

    while hi - lo >= precision:
        mid = (lo + hi) // 2
        if verbose:
            print(f"Trying n_facts = {mid} ...")
        success, combo = _evaluate_n_facts(
            d, n_attempts, mid, top_fraction_sweep, S_sweep,
            accuracy_threshold=accuracy_threshold,
            any_all_most=any_all_most, verbose=verbose,
        )
        if success:
            best, best_combo = mid, combo
            lo = mid + 1
            if verbose:
                print(f"✓  stored {mid} facts. Now searching: {lo} - {hi}\n")
        else:
            hi = mid - 1
            if verbose:
                print(f"✗  failed at {mid} facts. Now searching: {lo} - {hi}\n")

    # The loop never tests hi when it equals max_possible; check it explicitly.
    if hi == max_possible:
        if verbose:
            print(f"Trying n_facts = {max_possible} (max possible) ...")
        success, combo = _evaluate_n_facts(
            d, n_attempts, max_possible, top_fraction_sweep, S_sweep,
            accuracy_threshold=accuracy_threshold,
            any_all_most=any_all_most, verbose=verbose,
        )
        if success:
            best, best_combo = max_possible, combo

    if verbose:
        print(f"\nMax storable facts: {best}  (best top_fraction, S = {best_combo})")
    return best, best_combo


CAPACITY_RESULTS_PATH = os.path.join(RESULTS_DIR, "capacity_search_results.json")


def _append_capacity_result(max_facts, best_combo):
    """Append one JSON line summarising this run; never touches previous lines."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    best_tf, best_S = (best_combo if best_combo is not None else (None, None))
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "d": d,
        "max_facts": max_facts,
        "best_top_fraction": best_tf,
        "best_S": best_S,
        "accuracy_threshold": accuracy_threshold,
        "any_all_most": any_all_most,
        "n_attempts": n_attempts,
        "precision": precision,
        "S_sweep": S_sweep,
        "top_fraction_sweep": top_fraction_sweep,
    }
    with open(CAPACITY_RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return CAPACITY_RESULTS_PATH


@app.local_entrypoint()
def main():
    best, best_combo = find_max_facts(
        d=d,
        n_attempts=n_attempts,
        top_fraction_sweep=top_fraction_sweep,
        S_sweep=S_sweep,
        precision=precision,
        accuracy_threshold=accuracy_threshold,
        any_all_most=any_all_most,
        verbose=True,
    )
    print(f"\nd={d}: max_facts={best}, best (top_fraction, S)={best_combo}")
    path = _append_capacity_result(best, best_combo)
    print(f"Appended result to {path}")
