"""Capacity search for HandCodedModel2 (searches over integer top_n), on Modal.

Binary-search the largest n_facts that HandCodedModel2 can store, where "can
store" means: sweeping (S, top_n), some cell of that grid reaches
`accuracy_threshold` under the any/all/most rule.

Why top_n (not top_fraction): top_n is the number of top tokens (per input
position) a neuron suppresses — the actual knob. An experiment showed the optimal
top_n is small but scales with S (roughly top_n* <= ~2*S); a fixed top_fraction
grid maps to wildly different (and too-coarse) top_n across model sizes, so it
missed optima. We therefore search top_n as integers over a per-S range
0..round(2.2*S).

Efficiency: the connection matrix depends only on (D=d, T=d, S) — NOT on n_facts,
top_n, or attempt — so there is exactly ONE matrix per (d, S). We build them once
per d (in parallel on Modal), cache them to hand_coded_models/conn_cache/ on disk,
and reuse them across every n_facts probe, every top_n, every attempt, and every
config. Attempts share the base matrix and the facts; to add variety back, each
attempt permutes the matrix's label columns (mixing which labels share neurons,
structure preserved) and re-seeds the constructor's random tie-breaking.

Automation: set CONFIGS (a list of d x accuracy_threshold x any_all_most) at the
top; main() runs find_max_facts for each and appends one line per run to
hc2_sweep_results/capacity_search_results_topn.json (a SEPARATE log from the old
top_fraction runs; nothing old is overwritten). Grid files are written under new
names hc2_sweep_topn_d{d}_nfacts{nf}.json, leaving the old hc2_sweep_d*_nfacts*
files (and hc2_sweep_plot.py) untouched.

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
RESULTS_DIR = os.path.join(_HERE, "hc2_sweep_results")
# Top_n grid files live in their own subfolder; only the capacity_search_results_*
# logs sit directly in RESULTS_DIR. See hc2_sweep_results/README.md.
GRIDS_DIR = os.path.join(RESULTS_DIR, "topn_grids")
CONN_CACHE_DIR = os.path.join(_HERE, "conn_cache")


# ── Settings ──────────────────────────────────────────────────────────────────
n_attempts = 11

# Each n_facts grid is fanned out across ~this many Modal containers (the cap on
# simultaneous containers). Cells are sized so the (S x attempt x top_n) work
# splits into roughly this many balanced units.
TARGET_TASKS = 1000

# Each capacity search runs one (d, accuracy_threshold, any_all_most) config.
# This list reproduces the 24-cell sweep you were doing by hand, now automated.
CONFIGS = [
    dict(d=d, accuracy_threshold=thr, any_all_most=aam)
    for d in [16, 32, 64, 128, 256]
    for thr in [0.9, 1.0]
    for aam in ["any", "all", "most"]
]

testing = False  # small/cheap end-to-end validation run


def S_sweep_for(d):
    """Which n_neurons_per_label values to sweep for a given model size d."""
    if d <= 16:
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif d <= 32:
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]
    elif d <= 64:
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    elif d <= 128:
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    else:
        return list(range(1, 25))


def precision_for(d):
    """Binary-search stopping precision (stop when hi - lo < precision).

    d/2 matches the resolution the earlier top_fraction runs used, so the new
    top_n numbers are directly comparable. The previous 8/32/128/512 formula was
    too coarse for d>=64 and floored the small thr=1.0 capacities to 0."""
    return d // 2


def top_n_sweep_for(S):
    """Integer top_n grid for a given S: 0 .. round(2.2*S) (covers observed optima)."""
    return list(range(0, round(2.2 * S) + 1))


# All connection matrices use this fixed seed, so a given (d, S) maps to ONE matrix.
MATRIX_SEED = seed


def _tie_seed(S, attempt):
    """Per-(S, attempt) seed for the constructor's random tie-breaking — the only
    source of variation between attempts now that the matrix is shared per (d, S)."""
    return 10_000 * S + attempt


# ── Modal app + image ─────────────────────────────────────────────────────────
app = modal.App("hc2-capacity-search")

# Self-contained image: ships hc2.py + the repo-root modules it imports. No
# top-level import of `hand_coded_models` (Modal re-imports THIS file inside every
# container and that package is not on the container path).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "wandb")
    .add_local_python_source("models", "device")
    .add_local_file(os.path.join(_HERE, "hc2.py"), "/root/hc2.py")
)


@app.function(image=image, timeout=86400, max_containers=TARGET_TASKS)
def _build_conn(arg):
    """Build the ONE connection matrix for a (D, S). arg = (D, S). Returns (arg, matrix)."""
    import sys
    import warnings
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    from hc2 import make_connection_matrix

    D, S = arg
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = make_connection_matrix(D=D, T=D, S=S, seed=MATRIX_SEED)
    return (arg, m)


@app.function(image=image, timeout=86400, max_containers=TARGET_TASKS)
def _run_one(cell):
    """Evaluate one (S, attempt, n_facts) over its whole top_n grid, reusing the
    precomputed connection matrix carried in the cell. Returns a list of records.

    Per-attempt variety comes from permuting the matrix's label columns (and from
    re-seeding the constructor's tie-breaking) — no matrix is rebuilt."""
    import sys
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    import torch
    from hc2 import HandCodedModel2, HandCodedModel2Settings

    conn = cell["conn"]                 # np.ndarray (d_ff, n_labels), shared base matrix for this S
    S = cell["S"]
    attempt = cell["attempt"]
    n_facts = cell["n_facts"]
    tie_seed = cell["tie_seed"]

    # Per-attempt randomness without rebuilding: permute the label COLUMNS (axis 1
    # = output labels) of the shared matrix, so which labels share neurons differs
    # across attempts. A column permutation preserves all structural properties
    # (S ones per column, row sums, pairwise overlaps) — it only relabels.
    perm = np.random.default_rng(tie_seed).permutation(conn.shape[1])
    conn = conn[:, perm]

    records = []
    for top_n in cell["top_n_list"]:
        # Reproducible random tie-breaking inside the constructor (varies per attempt).
        torch.manual_seed(tie_seed + top_n)
        settings = HandCodedModel2Settings(
            input_vocab_size=cell["input_vocab_size"],
            output_vocab_size=cell["output_vocab_size"],
            n_facts=n_facts,
            seed=cell["seed"],
            d_ff=cell["d_ff"],
            n_neurons_per_label=S,
            use_top_no_top_fraction="top_n",
            top_n=top_n,
        )
        model = HandCodedModel2(settings, precomputed_conn=conn)
        _, best_guess_accuracy, _, _ = model.evaluate()
        records.append({
            "n_facts": n_facts,
            "S": S,
            "top_n": top_n,
            "attempt": attempt,
            "tie_seed": tie_seed,
            "best_guess_accuracy": best_guess_accuracy,
        })
    return records


# ── Connection-matrix cache (driver-side disk + Modal build) ──────────────────

def _conn_path(D, S):
    return os.path.join(CONN_CACHE_DIR, f"d{D}_s{S}.npy")


def _ensure_conn(d, S_sweep, verbose=True):
    """Return {S: matrix} for all S (one matrix per (d, S)), building the missing
    ones once on Modal and caching every matrix to disk under conn_cache/."""
    os.makedirs(CONN_CACHE_DIR, exist_ok=True)
    out = {}
    to_build = []          # list of (d, S)
    for S in S_sweep:
        p = _conn_path(d, S)
        if os.path.exists(p):
            out[S] = np.load(p)
        else:
            to_build.append((d, S))

    if to_build:
        if verbose:
            print(f"  Building {len(to_build)} connection matrices on Modal "
                  f"(d={d}, one per S); reused across all n_facts/top_n/attempts/configs ...")
        for (D, S), m in _build_conn.map(to_build):
            np.save(_conn_path(D, S), m)
            out[S] = m
    elif verbose:
        print(f"  All {len(out)} connection matrices for d={d} loaded from cache.")
    return out


# ── Result caching (grid files: hc2_sweep_topn_d{d}_nfacts{nf}.json) ───────────

def _grid_glob():
    return os.path.join(GRIDS_DIR, "hc2_sweep_topn_d*_nfacts*.json")


def _load_cached_records(d, n_facts):
    """Pool every top_n grid record for (d, n_facts) from disk."""
    records = []
    for path in sorted(glob.glob(_grid_glob())):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        s = payload.get("settings", {})
        if s.get("d") == d and s.get("n_facts") == n_facts:
            records.extend(payload["results"])
    return records


def _covers(records, S_sweep, n_attempts):
    """True if `records` already hold >= n_attempts runs for every (S, top_n) wanted."""
    counts = defaultdict(int)
    for r in records:
        counts[(r["S"], r["top_n"])] += 1
    for S in S_sweep:
        for tn in top_n_sweep_for(S):
            if counts[(S, tn)] < n_attempts:
                return False
    return True


def _build_cells_for_n_facts(d, n_facts, n_attempts, S_sweep, conn_by_S):
    """Fan a grid out into ~TARGET_TASKS balanced units for Modal.

    A unit is one (S, attempt) plus a CHUNK of that S's top_n grid. Splitting the
    top_n grid into chunks (rather than one big cell per (S, attempt)) keeps unit
    durations uniform — high-S cells no longer run 15x longer than low-S ones — so
    Modal can keep many more containers busy at once. All chunks of an (S, attempt)
    use the same per-attempt column permutation (it is derived from tie_seed)."""
    # Total (S, attempt, top_n) evaluations -> chunk size that hits ~TARGET_TASKS units.
    total = sum(n_attempts * len(top_n_sweep_for(S)) for S in S_sweep)
    chunk = max(1, round(total / TARGET_TASKS))

    cells = []
    for S in S_sweep:
        tns = top_n_sweep_for(S)
        conn = conn_by_S[S]
        for a in range(n_attempts):
            for i in range(0, len(tns), chunk):
                cells.append({
                    "conn": conn,
                    "S": S,
                    "attempt": a,
                    "n_facts": n_facts,
                    "tie_seed": _tie_seed(S, a),
                    "top_n_list": tns[i:i + chunk],
                    "input_vocab_size": 2 * d,
                    "output_vocab_size": d,
                    "d_ff": d,
                    "seed": seed,
                })
    return cells


def _save_records(d, n_facts, records, n_attempts, S_sweep):
    """Write a top_n grid result file (one per n_facts) into topn_grids/."""
    os.makedirs(GRIDS_DIR, exist_ok=True)
    payload = {
        "settings": {
            "d": d,
            "n_facts": n_facts,
            "n_attempts": n_attempts,
            "S_sweep": S_sweep,
            "input_vocab_size": 2 * d,
            "output_vocab_size": d,
            "d_ff": d,
            "seed": seed,
            "metric": "best_guess_accuracy",
            "search_mode": "top_n",
        },
        "results": records,
    }
    out_path = os.path.join(GRIDS_DIR, f"hc2_sweep_topn_d{d}_nfacts{n_facts}.json")
    base, ext = os.path.splitext(out_path)
    i = 1
    while os.path.exists(out_path):
        out_path = f"{base}_({i}){ext}"
        i += 1
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


# ── Grid evaluation ───────────────────────────────────────────────────────────

def _evaluate_n_facts(d, n_attempts, n_facts, S_sweep, conn_by_S,
                      accuracy_threshold=1.0, any_all_most="any", verbose=True):
    """Decide whether n_facts is storable, by sweeping (S, top_n).

    Returns (success, best_combination) where best_combination is the (top_n, S)
    whose any/all/most score is highest. Uses the disk cache if a covering grid
    already exists; otherwise runs the grid on Modal and saves it.
    """
    if any_all_most not in ("any", "all", "most"):
        raise ValueError("any_all_most must be 'any', 'all', or 'most'")

    records = _load_cached_records(d, n_facts)
    if _covers(records, S_sweep, n_attempts):
        if verbose:
            print(f"    n_facts={n_facts}: loaded grid from cache ({len(records)} records)")
    else:
        cells = _build_cells_for_n_facts(d, n_facts, n_attempts, S_sweep, conn_by_S)
        if verbose:
            print(f"    n_facts={n_facts}: running {len(cells)} cells on Modal ...")
        nested = list(_run_one.map(cells))
        records = [r for cell_records in nested for r in cell_records]
        path = _save_records(d, n_facts, records, n_attempts, S_sweep)
        if verbose:
            print(f"    n_facts={n_facts}: wrote {len(records)} records to {path}")

    # Collect each (S, top_n) cell's runs, restricted to the wanted grid.
    wanted = {(S, tn) for S in S_sweep for tn in top_n_sweep_for(S)}
    cells = defaultdict(list)
    for r in records:
        key = (r["S"], r["top_n"])
        if key in wanted:
            cells[key].append(r["best_guess_accuracy"])

    # Reduce each cell to one score matching the rule, so success <=> best >= threshold.
    #   any -> max over runs ; all -> min over runs ; most -> median over runs
    reduce_fn = {"any": np.max, "all": np.min, "most": np.median}[any_all_most]

    best_score = -1.0
    best_combination = None
    for (S, tn), accs in cells.items():
        score = float(reduce_fn(accs))
        if score > best_score:
            best_score = score
            best_combination = (tn, S)

    success = best_score >= accuracy_threshold
    if verbose and best_combination is not None:
        tn, S = best_combination
        print(f"    n_facts={n_facts}: best {any_all_most} cell (top_n={tn}, S={S}) "
              f"score={best_score:.4f} {'>=' if success else '<'} {accuracy_threshold} "
              f"-> {'PASS' if success else 'fail'}")
    return success, best_combination


# ── Binary search over n_facts ────────────────────────────────────────────────

def find_max_facts(d, n_attempts, S_sweep, precision=1,
                   accuracy_threshold=1.0, any_all_most="any", verbose=True):
    """Binary-search the maximum storable n_facts. max_possible = (2*d)^2 = 4*d**2.

    Returns (best_n_facts, best_combination) with best_combination = (top_n, S).
    """
    # Build/cache the connection matrices for this d ONCE (one per S); reused everywhere.
    conn_by_S = _ensure_conn(d, S_sweep, verbose=verbose)

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
            d, n_attempts, mid, S_sweep, conn_by_S,
            accuracy_threshold=accuracy_threshold,
            any_all_most=any_all_most, verbose=verbose,
        )
        if success:
            best, best_combo = mid, combo
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
        success, combo = _evaluate_n_facts(
            d, n_attempts, max_possible, S_sweep, conn_by_S,
            accuracy_threshold=accuracy_threshold,
            any_all_most=any_all_most, verbose=verbose,
        )
        if success:
            best, best_combo = max_possible, combo

    if verbose:
        print(f"\nMax storable facts: {best}  (best top_n, S = {best_combo})")
    return best, best_combo


# ── Result logging + driver ───────────────────────────────────────────────────
# Separate log from the old top_fraction runs (capacity_search_results.json) so
# the two approaches' results never mix and nothing old is overwritten.
CAPACITY_RESULTS_PATH = os.path.join(RESULTS_DIR, "capacity_search_results_topn.json")


def _append_capacity_result(d, max_facts, best_combo, accuracy_threshold,
                            any_all_most, n_attempts, precision, S_sweep):
    """Append one JSON line summarising a run; never touches previous lines."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    best_top_n, best_S = (best_combo if best_combo is not None else (None, None))
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "d": d,
        "max_facts": max_facts,
        "best_top_n": best_top_n,
        "best_S": best_S,
        "accuracy_threshold": accuracy_threshold,
        "any_all_most": any_all_most,
        "n_attempts": n_attempts,
        "precision": precision,
        "S_sweep": S_sweep,
        "search_mode": "top_n",
    }
    with open(CAPACITY_RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return CAPACITY_RESULTS_PATH


@app.local_entrypoint()
def main():
    configs = CONFIGS
    attempts = n_attempts
    if testing:
        # Cheap end-to-end validation: smallest d, few attempts, coarse precision.
        configs = [dict(d=16, accuracy_threshold=1.0, any_all_most="any")]
        attempts = 2

    print(f"Running {len(configs)} capacity search config(s).")
    for cfg in configs:
        d = cfg["d"]
        thr = cfg["accuracy_threshold"]
        aam = cfg["any_all_most"]
        S_sweep = S_sweep_for(d)
        precision = 256 if testing else precision_for(d)

        print(f"\n===== d={d}, accuracy_threshold={thr}, any_all_most={aam} =====")
        best, combo = find_max_facts(
            d, attempts, S_sweep, precision=precision,
            accuracy_threshold=thr, any_all_most=aam, verbose=True,
        )
        print(f"d={d}: max_facts={best}, best (top_n, S)={combo}")
        path = _append_capacity_result(
            d, best, combo, thr, aam, attempts, precision, S_sweep)
        print(f"Appended result to {path}")
