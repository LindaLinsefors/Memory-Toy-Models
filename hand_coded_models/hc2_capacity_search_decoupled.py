"""Capacity search for HandCodedModel2 with DECOUPLED model dimensions, on Modal.

Unlike hc2_capacity_search.py (which ties input_vocab_size = 2*d,
output_vocab_size = d, d_ff = d to a single `d`), this experiment sweeps
`input_vocab_size`, `output_vocab_size`, and `d_ff` independently. The knob is
`top_fraction` (a fixed grid for every S), as requested.

Parallelism structure
----------------------
- Each MODEL SETTING (input_vocab_size x output_vocab_size x d_ff) runs as ONE
  Modal container, and all model settings run in PARALLEL (`_run_model_setting.map`).
- WITHIN a model setting, the ACCURACY SETTINGS (accuracy_threshold x any/most/all)
  run in SERIAL. The (S, top_fraction) hyper-parameter grids used to decide
  storability depend only on (model setting, n_facts) — NOT on the accuracy
  setting — so every grid a container computes is reused across all six accuracy
  settings for that container (in-memory + on the grids Volume).

The connection matrix depends only on (D=d_ff, T=output_vocab_size, S) — not on
input_vocab_size, n_facts, top_fraction, or attempt — so there is exactly ONE
matrix per (d_ff, output_vocab_size, S). They are pre-built once (in parallel)
before the search and read from a Volume by every container.

File-I/O note (why a Volume is used for grids)
----------------------------------------------
Because the capacity search itself now runs inside parallel Modal containers, a
container cannot write grid files to the laptop's local *_grids folder during the
run. The fix (the pattern this codebase already uses for connection matrices):
grid files live on a Modal Volume during the run, keyed by
(input_vocab, output_vocab, d_ff, n_facts) so different containers write DISJOINT
filenames — no collision. The local entrypoint uploads any existing local grids to
the Volume before launching (so prior results are reused) and downloads all grids
back to the local topfrac_decoupled_grids/ folder when the run finishes. Nothing
old is overwritten and the local folder stays the durable source of truth.

Run it with:
    python -m modal run hand_coded_models/hc2_capacity_search_decoupled.py
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
S_sweep = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
top_frac_sweep = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 
                  0.20, 0.22, 0.24, 0.26, 0.28]

# Attempts per (S, top_fraction) cell; the reduction (any=max / most=median /
# all=min) is taken over these. Matches the other capacity searches.
n_attempts = 11

# Binary-search stopping precision (stop when hi - lo < PRECISION). With
# max_facts = input_vocab_size**2 (<= 4096 here) precision 1 is only ~12 probes,
# so we search exactly. Raise it if you want to trade resolution for speed.
PRECISION = 1

# Pass-through to HandCodedModel2 (the note in hc2.py says it changes nothing for
# the default down-connections; kept False here for the standard model).
add_possitive_down_connections = False

testing = False  # small/cheap end-to-end validation run


def s_sweep_for(d_ff):
    """S values valid for this d_ff. make_connection_matrix needs 0 < S <= D=d_ff,
    so S_sweep is clipped (e.g. d_ff=8 keeps S=1..8)."""
    return [S for S in S_sweep if S <= d_ff]


# All connection matrices use this fixed seed, so (d_ff, output_vocab, S) -> ONE matrix.
MATRIX_SEED = seed


def _tie_seed(S, attempt):
    """Per-(S, attempt) seed for the per-attempt column permutation and the
    constructor's random tie-breaking — the only source of variation between
    attempts now that the matrix is shared per (d_ff, output_vocab, S)."""
    return 10_000 * S + attempt


# ── Modal app + image ─────────────────────────────────────────────────────────
app = modal.App("hc2-capacity-search-decoupled")

# Self-contained image: ships hc2.py + the repo-root modules it imports.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "wandb")
    .add_local_python_source("models", "device")
    .add_local_file(os.path.join(_HERE, "hc2.py"), "/root/hc2.py")
)

# Connection matrices live on a Volume (one file per (d_ff, output_vocab, S)),
# pre-built before the search and read by every model-setting container.
conn_volume = modal.Volume.from_name("hc2-decoupled-conn-cache", create_if_missing=True)
CONN_VOL_MOUNT = "/conn_cache"

# Grid result files live on a separate Volume during the run; the local
# entrypoint syncs them to/from the local topfrac_decoupled_grids/ folder.
grids_volume = modal.Volume.from_name("hc2-decoupled-topfrac-grids", create_if_missing=True)
GRIDS_VOL_MOUNT = "/grids"

# Local mirror of the grids Volume (the durable *_grids folder the user asked for)
# and the high-level capacity log (JSONL, one line per model x accuracy setting).
LOCAL_GRIDS_DIR = os.path.join(RESULTS_DIR, "topfrac_decoupled_grids")
CAPACITY_RESULTS_PATH = os.path.join(RESULTS_DIR, "capacity_search_results_topfrac_decoupled.json")


def _conn_vol_name(d_ff, output_vocab, S):
    return f"dff{d_ff}_ov{output_vocab}_s{S}.npy"


def _grid_name(iv, ov, d_ff, n_facts):
    return f"hc2_topfrac_iv{iv}_ov{ov}_dff{d_ff}_nfacts{n_facts}.json"


def _grid_glob_pattern(iv, ov, d_ff):
    return f"hc2_topfrac_iv{iv}_ov{ov}_dff{d_ff}_nfacts*.json"


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


# ── One model setting = one parallel container ────────────────────────────────

@app.function(image=image, timeout=86400,
              volumes={CONN_VOL_MOUNT: conn_volume, GRIDS_VOL_MOUNT: grids_volume})
def _run_model_setting(cfg):
    """Run the FULL capacity search for one model setting (all accuracy settings,
    serially). cfg = {input_vocab_size, output_vocab_size, d_ff}.

    The 48 model settings run in parallel (one container each); within a container
    the binary search over n_facts is serial and each grid's (S, tf, attempt) cells
    are evaluated in a plain loop. Each cell is only a few ms, so keeping them
    in-container (rather than fanning out one tiny Modal task per cell) avoids huge
    per-task overhead and the resulting map backlog. Grids are read from / written
    to the grids Volume and cached in-process, so they are reused across every
    accuracy setting here and across future runs."""
    import os
    import sys
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    import torch
    from hc2 import HandCodedModel2, HandCodedModel2Settings

    iv = cfg["input_vocab_size"]
    ov = cfg["output_vocab_size"]
    d_ff = cfg["d_ff"]
    eff_S_sweep = s_sweep_for(d_ff)
    max_possible = iv ** 2

    # Load every connection matrix for this (d_ff, ov) once (one per S), reused for
    # all n_facts / tf / attempt evaluations. A warm container may hold a stale
    # Volume view of a matrix committed after it mounted; reload once and retry.
    conn_by_S = {}
    for S in eff_S_sweep:
        path = os.path.join(CONN_VOL_MOUNT, _conn_vol_name(d_ff, ov, S))
        try:
            conn_by_S[S] = np.load(path)
        except FileNotFoundError:
            conn_volume.reload()
            conn_by_S[S] = np.load(path)

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
        """True if records hold >= n_attempts runs for every (S, tf) wanted."""
        counts = defaultdict(int)
        for r in records:
            counts[(r["S"], r["top_fraction"])] += 1
        for S in eff_S_sweep:
            for tf in top_frac_sweep:
                if counts[(S, tf)] < n_attempts:
                    return False
        return True

    def _compute_grid(n_facts):
        """Evaluate every (S, tf, attempt) cell for n_facts, serially in-container."""
        records = []
        for S in eff_S_sweep:
            base_conn = conn_by_S[S]
            for attempt in range(n_attempts):
                tie_seed = _tie_seed(S, attempt)
                # Per-attempt variety without rebuilding the matrix: permute the
                # label COLUMNS of the shared matrix (structure-preserving relabel
                # of which labels share neurons).
                perm = np.random.default_rng(tie_seed).permutation(base_conn.shape[1])
                conn = base_conn[:, perm]
                for tf_index, tf in enumerate(top_frac_sweep):
                    # Reproducible tie-breaking in the constructor, varied per cell.
                    torch.manual_seed(tie_seed + tf_index)
                    settings = HandCodedModel2Settings(
                        input_vocab_size=iv,
                        output_vocab_size=ov,
                        n_facts=n_facts,
                        seed=seed,
                        d_ff=d_ff,
                        n_neurons_per_label=S,
                        use_top_no_top_fraction="top_fraction",
                        top_fraction=tf,
                        add_possitive_down_connections=add_possitive_down_connections,
                    )
                    model = HandCodedModel2(settings, precomputed_conn=conn)
                    _, best_guess_accuracy, _, _ = model.evaluate()
                    records.append({
                        "n_facts": n_facts,
                        "S": S,
                        "top_fraction": tf,
                        "attempt": attempt,
                        "tie_seed": tie_seed,
                        "best_guess_accuracy": best_guess_accuracy,
                    })
        return records

    def _save_grid_to_volume(n_facts, records):
        payload = {
            "settings": {
                "input_vocab_size": iv,
                "output_vocab_size": ov,
                "d_ff": d_ff,
                "n_facts": n_facts,
                "n_attempts": n_attempts,
                "S_sweep": eff_S_sweep,
                "top_frac_sweep": top_frac_sweep,
                "seed": seed,
                "metric": "best_guess_accuracy",
                "search_mode": "top_fraction",
                "add_possitive_down_connections": add_possitive_down_connections,
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
                "n_attempts": n_attempts,
                "precision": PRECISION,
                "S_sweep": eff_S_sweep,
                "top_frac_sweep": top_frac_sweep,
                "search_mode": "top_fraction",
                "add_possitive_down_connections": add_possitive_down_connections,
            })
    return results


# ── Local <-> Volume grid sync ────────────────────────────────────────────────

def _upload_local_grids():
    """Upload any grid files already in the local folder to the Volume, so a
    re-run reuses them instead of recomputing. Idempotent (force overwrite)."""
    local_files = sorted(glob.glob(os.path.join(LOCAL_GRIDS_DIR, "hc2_topfrac_*.json")))
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
    # No grids_volume.reload() here: this runs in the local entrypoint, where
    # reload() is disallowed AND unnecessary — iterdir/read_file query the server
    # fresh each call, so they already see the containers' latest commits.
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
    if testing:
        # Cheap end-to-end validation: one small model setting.
        ivs, ovs, dffs = [16], [8], [8]

    model_settings = [
        dict(input_vocab_size=iv, output_vocab_size=ov, d_ff=d_ff)
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
          f"setting(s) serially within each.")
    all_records = []
    for records in _run_model_setting.map(model_settings):
        all_records.extend(records)
        # Mirror grids to the laptop as each model setting finishes, so an
        # interrupted run still leaves a fresh local copy (only new files fetched).
        _download_grids_to_local(verbose=False)

    # Final sync (catches anything the per-setting syncs missed), then log.
    _download_grids_to_local()
    path = _append_capacity_results(all_records)
    print(f"Appended {len(all_records)} capacity record(s) to {path}")



