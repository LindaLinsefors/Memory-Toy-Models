"""Capacity search for HybridModel2 with DECOUPLED model dimensions, on Modal.

Mirror of hc2_capacity_search_decoupled.py, with the model swapped for
HybridModel2: the up matrix is generated exactly like HandCodedModel2's (and
frozen), while the down matrix + bias are randomly initialised and trained with
full-batch CE loss (plain Adam, lr=1e-2, up to 5000 epochs, early stopping —
see hc2_hybrid.py). A cell's score is the best best_guess_accuracy observed
during training. The knob is `top_fraction` (a fixed grid for every S), as in
the original.

Parallelism structure (one change from the original)
-----------------------------------------------------
- Each MODEL SETTING (input_vocab_size x output_vocab_size x d_ff) still runs
  as ONE Modal container, all in PARALLEL, with the ACCURACY SETTINGS serial
  within it and every grid reused across accuracy settings.
- WITHIN a model setting, a grid's (S, tf, attempt) cells are now TRAINING RUNS
  (up to 5000 epochs) instead of single forward passes. Computing them serially
  in-container — the original design — would take days, so the container fans
  them out to the `_train_attempt` worker function with `.map` (Modal supports
  calling Functions from other Functions). Each worker task covers one
  (S, attempt) x the WHOLE top_fraction sweep, and the maps use
  order_outputs=False — both are throughput fixes: Modal's .map only admits a
  bounded window of inputs at a time, so one-cell tasks left the container
  fleet far below the account limit, and ordered output let one slow cell
  stall the window. Grid caching on the grids Volume is unchanged (and cell
  seeding is identical), so completed grids are never retrained.

The connection matrices depend only on (d_ff, output_vocab, S, seed) — all
identical to the hand-coded decoupled search — so its Volume
(hc2-decoupled-conn-cache) is shared and nothing is rebuilt. Grids and the
capacity log go to SEPARATE hybrid-suffixed locations (Volume
hc2-hybrid-decoupled-topfrac-grids, local folder topfrac_decoupled_hybrid_grids/,
log capacity_search_results_topfrac_decoupled_hybrid.json) so they never mix
with the hand-coded runs.

Run it with:
    python -m modal run hand_coded_models/hc2_hybrid_capacity_search_decoupled.py
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


# ── Settings ──────────────────────────────────────────────────────────────────

# Model settings — swept independently (their product gives the parallel units).
input_vocab_size_list = [16, 32, 64]
output_vocab_size_list = [8, 16, 32, 64]
d_ff_list = [8, 16, 32, 64]

# Accuracy settings — run in serial within each model setting (grids are reused).
accuracy_threshold_list = [0.9, 1.0]
any_most_all_list = ["any", "most", "all"]

# Hyper parameters (the grid searched to decide whether an n_facts is storable).
S_sweep = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
top_frac_sweep = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,
                  0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38]

# Attempts per (S, top_fraction) cell; the reduction (any=max / most=median /
# all=min) is taken over these. Matches the other capacity searches.
n_attempts = 11

# Training recipe for the down matrix (see hc2_hybrid.train_down_matrix).
N_EPOCHS = 5000
LR = 1e-2
PATIENCE = 100

# Binary-search stopping precision (stop when hi - lo < PRECISION).
PRECISION = 1

testing = False  # small/cheap end-to-end validation run


def s_sweep_for(d_ff):
    """S values valid for this d_ff. make_connection_matrix needs 0 < S <= D=d_ff,
    so S_sweep is clipped (e.g. d_ff=8 keeps S=1..8)."""
    return [S for S in S_sweep if S <= d_ff]


# All connection matrices use this fixed seed, so (d_ff, output_vocab, S) -> ONE matrix.
MATRIX_SEED = seed


def _tie_seed(S, attempt):
    """Per-(S, attempt) seed for the per-attempt column permutation and the
    constructor's random tie-breaking."""
    return 10_000 * S + attempt


# ── Modal app + image ─────────────────────────────────────────────────────────
app = modal.App("hc2-hybrid-capacity-search-decoupled")

# Self-contained image: ships hc2.py + hc2_hybrid.py + the repo-root modules
# they import.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "wandb")
    .add_local_python_source("models", "device")
    .add_local_file(os.path.join(_HERE, "hc2.py"), "/root/hc2.py")
    .add_local_file(os.path.join(_HERE, "hc2_hybrid.py"), "/root/hc2_hybrid.py")
)

# Connection matrices live on the SAME Volume as the hand-coded decoupled search
# (the matrices are identical), pre-built before the search and read by every
# training worker.
conn_volume = modal.Volume.from_name("hc2-decoupled-conn-cache", create_if_missing=True)
CONN_VOL_MOUNT = "/conn_cache"

# Grid result files live on their own (hybrid) Volume during the run; the local
# entrypoint syncs them to/from the local topfrac_decoupled_hybrid_grids/ folder.
grids_volume = modal.Volume.from_name("hc2-hybrid-decoupled-topfrac-grids", create_if_missing=True)
GRIDS_VOL_MOUNT = "/grids"

# Local mirror of the grids Volume (the durable *_grids folder) and the
# high-level capacity log (JSONL, one line per model x accuracy setting).
LOCAL_GRIDS_DIR = os.path.join(RESULTS_DIR, "topfrac_decoupled_hybrid_grids")
CAPACITY_RESULTS_PATH = os.path.join(RESULTS_DIR, "capacity_search_results_topfrac_decoupled_hybrid.json")


def _conn_vol_name(d_ff, output_vocab, S):
    return f"dff{d_ff}_ov{output_vocab}_s{S}.npy"


def _grid_name(iv, ov, d_ff, n_facts):
    return f"hc2_hybrid_topfrac_iv{iv}_ov{ov}_dff{d_ff}_nfacts{n_facts}.json"


def _grid_glob_pattern(iv, ov, d_ff):
    return f"hc2_hybrid_topfrac_iv{iv}_ov{ov}_dff{d_ff}_nfacts*.json"


# ── Connection-matrix build (Modal) ───────────────────────────────────────────

@app.function(image=image, timeout=86400, volumes={CONN_VOL_MOUNT: conn_volume})
def _build_conn(arg):
    """Build the ONE connection matrix for (d_ff, output_vocab, S) and persist it
    to the Volume. Idempotent: returns immediately if the file already exists.
    arg = (d_ff, output_vocab, S)."""
    import os
    import sys
    import warnings
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    from hc2 import make_connection_matrix

    d_ff, output_vocab, S = arg
    path = os.path.join(CONN_VOL_MOUNT, _conn_vol_name(d_ff, output_vocab, S))
    if os.path.exists(path):
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = make_connection_matrix(D=d_ff, T=output_vocab, S=S, seed=MATRIX_SEED)
    np.save(path, m)
    conn_volume.commit()


# ── One (S, attempt) = one worker task (trains the whole top_fraction sweep) ──

@app.function(image=image, timeout=86400, volumes={CONN_VOL_MOUNT: conn_volume})
def _train_attempt(cell):
    """Train + evaluate one (S, attempt) for a given n_facts, across ALL 20
    top_fraction values. Returns a LIST of records (one per top_fraction).

    Batching the tf sweep into one task (instead of one task per cell) matters
    for throughput: Modal's .map only admits a bounded window of inputs at a
    time, so with tiny one-cell tasks the fleet was stuck far below the account
    container limit. One task per (S, attempt) makes each admitted input carry
    20 training runs, and the per-task overhead is paid once.

    Each cell's model is IDENTICAL to the old one-task-per-cell version: the
    per-attempt column permutation depends only on tie_seed, and each tf uses
    torch.manual_seed(tie_seed + tf_index) and init_seed=tie_seed + tf_index,
    exactly as before — so grids cached by earlier runs stay consistent."""
    import os
    import sys
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    import torch
    from hc2 import HandCodedModel2Settings
    from hc2_hybrid import HybridModel2, train_down_matrix

    iv = cell["input_vocab_size"]
    ov = cell["output_vocab_size"]
    d_ff = cell["d_ff"]
    S = cell["S"]
    path = os.path.join(CONN_VOL_MOUNT, _conn_vol_name(d_ff, ov, S))
    try:
        conn = np.load(path)
    except FileNotFoundError:
        conn_volume.reload()
        conn = np.load(path)

    attempt = cell["attempt"]
    n_facts = cell["n_facts"]
    tie_seed = cell["tie_seed"]

    # Per-attempt variety without rebuilding the matrix: permute the label
    # COLUMNS of the shared matrix (structure-preserving relabel).
    perm = np.random.default_rng(tie_seed).permutation(conn.shape[1])
    conn = conn[:, perm]

    records = []
    for tf_index, tf in enumerate(top_frac_sweep):
        # Reproducible tie-breaking in the constructor, varied per cell.
        torch.manual_seed(tie_seed + tf_index)
        settings = HandCodedModel2Settings(
            input_vocab_size=iv,
            output_vocab_size=ov,
            n_facts=n_facts,
            seed=cell["seed"],
            d_ff=d_ff,
            n_neurons_per_label=S,
            use_top_no_top_fraction="top_fraction",
            top_fraction=tf,
        )
        model = HybridModel2(settings, precomputed_conn=conn,
                             init_seed=tie_seed + tf_index)
        best_guess_accuracy, epochs_run = train_down_matrix(
            model, n_epochs=N_EPOCHS, lr=LR, patience=PATIENCE)
        records.append({
            "n_facts": n_facts,
            "S": S,
            "top_fraction": tf,
            "attempt": attempt,
            "tie_seed": tie_seed,
            "best_guess_accuracy": best_guess_accuracy,
            "epochs_run": epochs_run,
        })
    return records


# ── One model setting = one parallel container ────────────────────────────────

@app.function(image=image, timeout=86400,
              volumes={CONN_VOL_MOUNT: conn_volume, GRIDS_VOL_MOUNT: grids_volume})
def _run_model_setting(cfg):
    """Run the FULL capacity search for one model setting (all accuracy settings,
    serially). cfg = {input_vocab_size, output_vocab_size, d_ff, n_attempts}.

    The model settings run in parallel (one container each); within a container
    the binary search over n_facts is serial, and each grid's (S, tf, attempt)
    TRAINING cells are fanned out to the _train_attempt worker with .map (a serial
    in-container loop, as in the hand-coded original, would take days now that
    a cell is a training run). Grids are read from / written to the grids
    Volume and cached in-process, so they are reused across every accuracy
    setting here and across future runs."""
    import os
    import sys
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    iv = cfg["input_vocab_size"]
    ov = cfg["output_vocab_size"]
    d_ff = cfg["d_ff"]
    attempts = cfg["n_attempts"]
    eff_S_sweep = s_sweep_for(d_ff)
    max_possible = iv ** 2

    grid_cache = {}  # n_facts -> flat list of per-(S, tf, attempt) records

    def _load_grid_from_volume(n_facts):
        """Pool every grid record for (this model setting, n_facts) from the Volume."""
        pattern = os.path.join(GRIDS_VOL_MOUNT, _grid_glob_pattern(iv, ov, d_ff))
        records = []
        for gp in sorted(glob.glob(pattern)):
            with open(gp, "r", encoding="utf-8") as f:
                payload = json.load(f)
            s = payload.get("settings", {})
            if (s.get("input_vocab_size") == iv and s.get("output_vocab_size") == ov
                    and s.get("d_ff") == d_ff and s.get("n_facts") == n_facts):
                records.extend(payload["results"])
        return records

    def _covers(records):
        """True if records hold >= attempts runs for every (S, tf) wanted."""
        counts = defaultdict(int)
        for r in records:
            counts[(r["S"], r["top_fraction"])] += 1
        for S in eff_S_sweep:
            for tf in top_frac_sweep:
                if counts[(S, tf)] < attempts:
                    return False
        return True

    def _compute_grid(n_facts):
        """Train every (S, tf, attempt) cell for n_facts, in parallel via
        _train_attempt: one task per (S, attempt), each covering the whole tf
        sweep (see _train_attempt for why). order_outputs=False streams results
        back as they finish, so one slow attempt can't stall input admission."""
        cells = []
        for S in eff_S_sweep:
            for attempt in range(attempts):
                cells.append({
                    "input_vocab_size": iv,
                    "output_vocab_size": ov,
                    "d_ff": d_ff,
                    "S": S,
                    "attempt": attempt,
                    "n_facts": n_facts,
                    "tie_seed": _tie_seed(S, attempt),
                    "seed": seed,
                })
        records = []
        for batch in _train_attempt.map(cells, order_outputs=False):
            records.extend(batch)
        return records

    def _save_grid_to_volume(n_facts, records):
        payload = {
            "settings": {
                "input_vocab_size": iv,
                "output_vocab_size": ov,
                "d_ff": d_ff,
                "n_facts": n_facts,
                "n_attempts": attempts,
                "S_sweep": eff_S_sweep,
                "top_frac_sweep": top_frac_sweep,
                "seed": seed,
                "metric": "best_guess_accuracy",
                "search_mode": "top_fraction",
                "model": "HybridModel2",
                "n_epochs": N_EPOCHS,
                "lr": LR,
                "patience": PATIENCE,
            },
            "results": records,
        }
        out_path = os.path.join(GRIDS_VOL_MOUNT, _grid_name(iv, ov, d_ff, n_facts))
        base, ext = os.path.splitext(out_path)
        i = 1
        while os.path.exists(out_path):
            out_path = f"{base}_({i}){ext}"
            i += 1
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        grids_volume.commit()

    def _get_grid(n_facts):
        """Records for n_facts: in-memory -> Volume -> compute (then persist)."""
        if n_facts in grid_cache:
            return grid_cache[n_facts]
        records = _load_grid_from_volume(n_facts)
        if not _covers(records):
            records = _compute_grid(n_facts)
            _save_grid_to_volume(n_facts, records)
        grid_cache[n_facts] = records
        return records

    def _storable(n_facts, accuracy_threshold, any_all_most):
        """Return (success, (best_top_fraction, best_S)) for this accuracy setting."""
        records = _get_grid(n_facts)
        wanted = {(S, tf) for S in eff_S_sweep for tf in top_frac_sweep}
        cells = defaultdict(list)
        for r in records:
            key = (r["S"], r["top_fraction"])
            if key in wanted:
                cells[key].append(r["best_guess_accuracy"])
        # any -> max over runs ; all -> min ; most -> median
        reduce_fn = {"any": np.max, "all": np.min, "most": np.median}[any_all_most]
        best_score, best_combo = -1.0, None
        for (S, tf), accs in cells.items():
            score = float(reduce_fn(accs))
            if score > best_score:
                best_score, best_combo = score, (tf, S)
        return best_score >= accuracy_threshold, best_combo

    def _find_max_facts(accuracy_threshold, any_all_most):
        """Binary-search the largest storable n_facts for one accuracy setting."""
        lo, hi = 1, max_possible
        best, best_combo = 0, None
        while hi - lo >= PRECISION:
            mid = (lo + hi) // 2
            success, combo = _storable(mid, accuracy_threshold, any_all_most)
            if success:
                best, best_combo = mid, combo
                lo = mid + 1
            else:
                hi = mid - 1
        if hi == max_possible:  # loop never tests the top end
            success, combo = _storable(max_possible, accuracy_threshold, any_all_most)
            if success:
                best, best_combo = max_possible, combo
        return best, best_combo

    results = []
    for thr in accuracy_threshold_list:
        for aam in any_most_all_list:
            best, combo = _find_max_facts(thr, aam)
            best_tf, best_S = (combo if combo is not None else (None, None))
            results.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "input_vocab_size": iv,
                "output_vocab_size": ov,
                "d_ff": d_ff,
                "max_facts": best,
                "best_top_fraction": best_tf,
                "best_S": best_S,
                "accuracy_threshold": thr,
                "any_all_most": aam,
                "n_attempts": attempts,
                "precision": PRECISION,
                "S_sweep": eff_S_sweep,
                "top_frac_sweep": top_frac_sweep,
                "search_mode": "top_fraction",
                "model": "HybridModel2",
                "n_epochs": N_EPOCHS,
                "lr": LR,
                "patience": PATIENCE,
            })
    return results


# ── Local <-> Volume grid sync ────────────────────────────────────────────────

def _upload_local_grids():
    """Upload any grid files already in the local folder to the Volume, so a
    re-run reuses them instead of recomputing. Idempotent (force overwrite)."""
    local_files = sorted(glob.glob(os.path.join(LOCAL_GRIDS_DIR, "hc2_hybrid_topfrac_*.json")))
    if not local_files:
        return
    print(f"Uploading {len(local_files)} existing local grid(s) to the Volume for reuse ...")
    with grids_volume.batch_upload(force=True) as batch:
        for p in local_files:
            batch.put_file(p, os.path.basename(p))


def _download_grids_to_local(verbose=True):
    """Mirror grid files from the Volume into the local folder.

    Only files not already present locally are fetched, so calling this
    repeatedly (e.g. after every model setting finishes) stays cheap. Grid files
    are immutable once written, so skipping existing names never misses an update."""
    os.makedirs(LOCAL_GRIDS_DIR, exist_ok=True)
    n = 0
    for entry in grids_volume.iterdir("/"):
        name = os.path.basename(entry.path)
        if not name.endswith(".json"):
            continue
        dest = os.path.join(LOCAL_GRIDS_DIR, name)
        if os.path.exists(dest):
            continue
        with open(dest, "wb") as f:
            for chunk in grids_volume.read_file(entry.path):
                f.write(chunk)
        n += 1
    if verbose:
        print(f"Synced {n} new grid file(s) from the Volume into {LOCAL_GRIDS_DIR}")


def _append_capacity_results(records):
    """Append the capacity records (JSONL); never touches previous lines."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CAPACITY_RESULTS_PATH, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return CAPACITY_RESULTS_PATH


# ── Driver ────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    ivs, ovs, dffs = input_vocab_size_list, output_vocab_size_list, d_ff_list
    attempts = n_attempts
    if testing:
        # Cheap end-to-end validation: one small model setting, few attempts.
        ivs, ovs, dffs = [16], [8], [8]
        attempts = 2

    model_settings = [
        dict(input_vocab_size=iv, output_vocab_size=ov, d_ff=d_ff, n_attempts=attempts)
        for iv in ivs for ov in ovs for d_ff in dffs
    ]

    # Pre-build every unique (d_ff, output_vocab, S) connection matrix once, in
    # parallel. Doing it in a separate phase (not inside the search containers)
    # avoids two containers racing to build the same matrix.
    conn_keys = sorted({
        (d_ff, ov, S)
        for iv in ivs for ov in ovs for d_ff in dffs
        for S in s_sweep_for(d_ff)
    })
    print(f"Ensuring {len(conn_keys)} connection matrices on the Volume ...")
    list(_build_conn.map(conn_keys))

    # Reuse prior grids: push the local folder up to the Volume before launching.
    _upload_local_grids()

    print(f"Running {len(model_settings)} model setting(s) in parallel; "
          f"{len(accuracy_threshold_list) * len(any_most_all_list)} accuracy "
          f"setting(s) serially within each; training fans out to _train_attempt.")
    n_total = 0
    n_done = 0
    # order_outputs=False: handle each model setting as it finishes (so the log
    # and grids update promptly) instead of waiting on the slowest-first one.
    for records in _run_model_setting.map(model_settings, order_outputs=False):
        # Append this setting's capacity rows NOW (JSONL append, one line per
        # accuracy setting), so progress is visible in the log while the run
        # goes — like the coupled search. An interrupted run keeps every
        # setting finished so far (its grids are on the Volume regardless).
        _append_capacity_results(records)
        n_total += len(records)
        n_done += 1
        r = records[0]
        print(f"[{n_done}/{len(model_settings)}] finished iv={r['input_vocab_size']}, "
              f"ov={r['output_vocab_size']}, d_ff={r['d_ff']} — appended {len(records)} rows")
        # Mirror grids to the laptop as each model setting finishes, so an
        # interrupted run still leaves a fresh local copy (only new files fetched).
        _download_grids_to_local(verbose=False)

    # Final sync (catches anything the per-setting syncs missed).
    _download_grids_to_local()
    print(f"Appended {n_total} capacity record(s) to {CAPACITY_RESULTS_PATH}")
