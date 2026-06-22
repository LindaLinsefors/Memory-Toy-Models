"""Sweep HandCodedModel2 over n_facts, top_fraction and n_neurons_per_label (S), on Modal.

Run it with:
    python -m modal run hand_coded_models/hc2_sweep.py

For the fixed d below, this evaluates HandCodedModel2 for every combination of
(n_facts, top_fraction, S), repeating each S `n_attempts` times. The facts are
held fixed (generate_facts always uses seed=42), but the connection matrix is
re-generated for every attempt with a distinct seed, so each attempt is a
genuinely different random construction.

Parallel structure: the work is fanned out as widely as possible — ONE container
per single evaluation (n_facts, S, top_fraction, attempt). Each builds its own
fresh connection matrix from a unique seed (we deliberately do NOT reuse a matrix
across evaluations: regenerating gives more variety, and Modal's parallelism
makes the extra simulated-annealing runs cheap in wall-clock time):

    main()  (your laptop)
      └── _run_one.map(over every (n_facts, S, top_fraction, attempt))
              builds a fresh connection matrix, evaluates one model, returns one record.

Each record is tagged with its n_facts; the driver groups them by n_facts and
writes ONE JSON file per n_facts under hand_coded_models/hc2_sweep_results/ (so
each file stays backwards compatible with the single-n_facts format and with
hc2_sweep_plot.py).
"""

import os
import json
import time
from datetime import datetime

import modal

# --- Settings -----------------------------------------------------------------
d = 128

n_attempts = 11

n_facts_sweep = [128, 256, 512, 1024, 2048, 4096, 8192]  # used for n_facts
top_fraction_sweep = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
S_sweep = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # used for n_neurons_per_label

input_vocab_size = 2 * d
output_vocab_size = d
d_ff = d
seed = 42  # fixed seed for generate_facts (the facts never change)

testing = False  # small/cheap run to validate the pipeline first

# --- Modal app + image --------------------------------------------------------
app = modal.App("hc2-sweep")

# hc2.py lives next to this file; ship it (plus the repo-root modules it imports)
# into the container so `import hc2` works there. The work is pure-CPU (simulated
# annealing + small tensor ops), so no GPU is requested.
_HERE = os.path.dirname(os.path.abspath(__file__))
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "wandb")
    .add_local_python_source("models", "device")
    .add_local_file(os.path.join(_HERE, "hc2.py"), "/root/hc2.py")
)


@app.function(image=image, timeout=86400)
def _run_one(args: dict) -> dict:
    """Evaluate ONE model: a single (n_facts, S, top_fraction, attempt) point.

    Builds its own fresh connection matrix from a unique conn_seed (no reuse — this
    is what gives variety across attempts). generate_facts inside HandCodedModel2
    uses the fixed `seed`, so for a given n_facts the facts are identical across
    attempts; only the connection matrix (and the random tie-breaking) varies.
    Returns one record dict.
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

    # Make the random tie-breaking inside the constructor reproducible per cell.
    torch.manual_seed(conn_seed)

    # Fresh connection matrix for this exact evaluation. Suppress the "T exceeds
    # theoretical maximum" warning — overlap violations are expected and SA
    # minimises them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conn = make_connection_matrix(D=args["d_ff"], T=args["output_vocab_size"],
                                      S=S, seed=conn_seed)

    settings = HandCodedModel2Settings(
        input_vocab_size=args["input_vocab_size"],
        output_vocab_size=args["output_vocab_size"],
        n_facts=args["n_facts"],
        seed=args["seed"],            # fixed → facts are identical every time
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


def _build_cells():
    """One work unit per (n_facts, S, top_fraction, attempt). Every unit gets a
    unique conn_seed so its connection matrix is independently regenerated."""
    s_values = [1, 2] if testing else S_sweep
    attempts = 2 if testing else n_attempts
    nf_values = n_facts_sweep[:2] if testing else n_facts_sweep
    cells = []
    idx = 0
    for nf in nf_values:
        for S in s_values:
            for top_fraction in top_fraction_sweep:
                for attempt in range(attempts):
                    cells.append({
                        "n_facts": nf,
                        "S": S,
                        "top_fraction": top_fraction,
                        "attempt": attempt,
                        # Unique per cell → a distinct, independently generated matrix.
                        "conn_seed": seed + 1 + idx,
                        "d_ff": d_ff,
                        "seed": seed,
                        "input_vocab_size": input_vocab_size,
                        "output_vocab_size": output_vocab_size,
                    })
                    idx += 1
    return cells


@app.local_entrypoint()
def main():
    # Grid files live in the top_fraction_grids/ subfolder (see
    # hc2_sweep_results/README.md); only capacity logs sit in the results root.
    results_dir = os.path.join(_HERE, "hc2_sweep_results", "top_fraction_grids")
    os.makedirs(results_dir, exist_ok=True)

    cells = _build_cells()
    nf_values = n_facts_sweep[:2] if testing else n_facts_sweep
    print(f"Launching {len(cells)} evaluations on Modal, one container task each "
          f"({len(nf_values)} n_facts x {len(S_sweep)} S x "
          f"{len(top_fraction_sweep)} top_fractions x {n_attempts} attempts), "
          f"each building its own connection matrix.\n")

    start = time.perf_counter()
    results = list(_run_one.map(cells))
    elapsed = time.perf_counter() - start
    print(f"Collected {len(results)} records in {elapsed:.1f}s.")

    # Group records by n_facts and write one file per n_facts. Each file keeps the
    # single-n_facts schema (so it stays backwards compatible with prior runs and
    # with hc2_sweep_plot.py).
    timestamp = datetime.now().isoformat(timespec="seconds")
    prefix = "test_" if testing else ""
    for nf in nf_values:
        nf_results = [r for r in results if r["n_facts"] == nf]
        payload = {
            "settings": {
                "d": d,
                "n_facts": nf,
                "n_attempts": n_attempts,
                "top_fraction_sweep": top_fraction_sweep,
                "S_sweep": S_sweep,
                "input_vocab_size": input_vocab_size,
                "output_vocab_size": output_vocab_size,
                "d_ff": d_ff,
                "seed": seed,
                "metric": "best_guess_accuracy",
            },
            "timestamp": timestamp,
            "results": nf_results,
        }

        out_path = os.path.join(results_dir, f"{prefix}hc2_sweep_d{d}_nfacts{nf}.json")
        # Make the path unique to avoid clobbering previous runs.
        base, ext = os.path.splitext(out_path)
        i = 1
        while os.path.exists(out_path):
            out_path = f"{base}_({i}){ext}"
            i += 1

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"  n_facts={nf}: wrote {len(nf_results)} records to {out_path}")
