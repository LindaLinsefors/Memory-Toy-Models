#%%
"""Visualise mlp_in / mlp_out (up_matrix / down_matrix) for hand-coded (hc2)
vs fully-trained networks at model size d=16.

For each success criterion (accuracy_threshold + any/all/most) we show three
d=16 networks:
  1. hand-coded  HandCodedModel2  at the hand-coded max_facts, with that
     criterion's best (S, top_fraction) read from the capacity logs;
  2. trained     FullyTrainedModel2 at the *trained* max_facts (the most it can learn);
  3. trained     FullyTrainedModel2 at the *hand-coded* max_facts (same fact count
     as #1, for a like-for-like comparison).

For every network we build/train the same 11 attempts the capacity search uses,
then keep the attempt matching that criterion's reduction:
  any -> best attempt, most -> median attempt, all -> worst attempt
(by final best_guess_accuracy). Trained nets use NO early stopping — they run the
full n_epochs and we visualise the final-epoch weights.

Nothing is read from saved weights (none exist); the hand-coded nets are rebuilt
and the trained nets are retrained here. Only scalar hyper-parameters come from
the logs. Figures are written to hand_coded_models/weight_viz/.
"""

import os
import sys
import json

# This script lives in hand_coded_models/; it imports sibling modules (hc2,
# hc2_full_train) AND repo-root modules (models, device). Put both on the path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import matplotlib.pyplot as plt

import models  # noqa: F401  -- importing sets the default torch device (cuda/cpu)
from hc2 import HandCodedModel2, HandCodedModel2Settings
from hc2_full_train import FullyTrainedModel2, train_full_network


RESULTS_DIR = os.path.join(_HERE, "hc2_sweep_results")
TOPFRAC_LOG = os.path.join(RESULTS_DIR, "capacity_search_results_topfrac.json")
FULLTRAIN_LOG = os.path.join(RESULTS_DIR, "capacity_search_results_fulltrain.json")
OUT_DIR = os.path.join(_HERE, "weight_viz")
# Trained nets are saved here so plot edits reuse the weights instead of retraining.
CACHE_DIR = os.path.join(OUT_DIR, "model_cache")

# Default torch device (models.py set it on import); saved-cpu weights load onto it.
DEV = torch.empty(0).device

D = 16
N_ATTEMPTS = 11
N_EPOCHS = 5000
LR = 1e-2

# The three success criteria the user asked for ("all of them").
CRITERIA = [
    (1.0, "any"),
    (0.9, "any"),
    (1.0, "most"),
]


# ── Log reading ───────────────────────────────────────────────────────────────

def _load_jsonl(path):
    """Read a JSONL capacity log into a list of dicts (file/append order)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _latest(path, d, thr, aam):
    """Latest (last-appended) row matching (d, accuracy_threshold, any_all_most)."""
    match = [r for r in _load_jsonl(path)
             if r.get("d") == d
             and r.get("accuracy_threshold") == thr
             and r.get("any_all_most") == aam]
    if not match:
        raise LookupError(f"No row for d={d}, thr={thr}, aam={aam} in {os.path.basename(path)}")
    return match[-1]


# ── Attempt selection per criterion ───────────────────────────────────────────

def _select(models_and_accs, aam):
    """Pick the representative attempt for a criterion's any/all/most rule.

    models_and_accs: list of (model, best_guess_accuracy).
      any  -> highest accuracy (the best one)
      most -> the median-accuracy attempt (representative, not the lucky best)
      all  -> lowest accuracy (worst attempt)
    Returns (model, accuracy, rank_note).
    """
    ordered = sorted(models_and_accs, key=lambda ma: ma[1])  # ascending accuracy
    if aam == "any":
        model, acc = ordered[-1]
        note = "best of %d" % len(ordered)
    elif aam == "all":
        model, acc = ordered[0]
        note = "worst of %d" % len(ordered)
    else:  # "most" -> median
        model, acc = ordered[len(ordered) // 2]
        note = "median of %d" % len(ordered)
    return model, acc, note


# ── Model builders (11 attempts each) ─────────────────────────────────────────

def build_handcoded(d, n_facts, S, top_fraction, aam):
    """Build N_ATTEMPTS HandCodedModel2 nets and return the criterion's representative."""
    attempts = []
    for a in range(N_ATTEMPTS):
        # Seed the global RNG so the constructor's randperm tie-breaking gives
        # 11 distinct, reproducible builds (the conn matrix itself is fixed by seed=42).
        torch.manual_seed(a)
        settings = HandCodedModel2Settings(
            input_vocab_size=2 * d,
            output_vocab_size=d,
            d_ff=d,
            n_facts=n_facts,
            n_neurons_per_label=S,
            use_top_no_top_fraction="top_fraction",
            top_fraction=top_fraction,
            seed=42,
        )
        model = HandCodedModel2(settings)
        _, bga, _, _ = model.evaluate()
        attempts.append((model, bga))
    return _select(attempts, aam)


# Trained nets are expensive; cache the 11 attempts per n_facts (fact counts are
# shared across criteria) both in-memory and on disk, so each count is trained at
# most once — ever. Plot edits reload the saved weights instead of retraining.
_trained_cache = {}


def _cache_path(n_facts):
    return os.path.join(CACHE_DIR, f"trained_d{D}_nfacts{n_facts}.pt")


def _rebuild_trained(d, n_facts, init_seed, up, down, bias):
    """Reconstruct a FullyTrainedModel2 (facts are deterministic from seed/n_facts)
    and load saved weights into it, so it forwards/plots exactly as trained."""
    model = FullyTrainedModel2(
        input_vocab_size=2 * d, output_vocab_size=d, d_ff=d,
        n_facts=n_facts, seed=42, init_seed=init_seed)
    model.up_matrix = up.to(DEV)
    model.down_matrix = down.to(DEV)
    model.down_bias = bias.to(DEV)
    return model


def train_trained(d, n_facts, aam):
    """Return the criterion's representative among N_ATTEMPTS trained nets for
    n_facts. Loads the 11 attempts from disk if cached; otherwise trains them (no
    early stopping) and saves them for reuse."""
    if n_facts not in _trained_cache:
        path = _cache_path(n_facts)
        if os.path.exists(path):
            blob = torch.load(path, map_location="cpu")
            attempts = [
                (_rebuild_trained(d, n_facts, r["init_seed"],
                                  r["up_matrix"], r["down_matrix"], r["down_bias"]),
                 r["bga"])
                for r in blob["attempts"]
            ]
            print(f"      trained n_facts={n_facts}: loaded {len(attempts)} "
                  f"attempts from cache")
        else:
            attempts, records = [], []
            for a in range(N_ATTEMPTS):
                model = FullyTrainedModel2(
                    input_vocab_size=2 * d, output_vocab_size=d, d_ff=d,
                    n_facts=n_facts, seed=42, init_seed=a)
                train_full_network(model, n_epochs=N_EPOCHS, lr=LR,
                                   early_stopping=False)
                _, bga, _, _ = model.evaluate()
                attempts.append((model, bga))
                records.append({
                    "init_seed": a, "bga": bga,
                    "up_matrix": model.up_matrix.detach().cpu(),
                    "down_matrix": model.down_matrix.detach().cpu(),
                    "down_bias": model.down_bias.detach().cpu(),
                })
                print(f"      trained n_facts={n_facts} attempt {a}: bga={bga:.4f}")
            os.makedirs(CACHE_DIR, exist_ok=True)
            torch.save({"d": d, "n_facts": n_facts, "n_epochs": N_EPOCHS,
                        "lr": LR, "attempts": records}, _cache_path(n_facts))
            print(f"      saved {N_ATTEMPTS} trained nets -> {_cache_path(n_facts)}")
        _trained_cache[n_facts] = attempts
    return _select(_trained_cache[n_facts], aam)


# ── Plotting ──────────────────────────────────────────────────────────────────

# Diverging colormap with RED = NEGATIVE, blue = positive (coolwarm reversed).
CMAP = "coolwarm_r"


def _mat(model, name):
    return getattr(model, name).detach().cpu().float().numpy()


def _activations(model):
    """Hidden-layer activations for every fact: relu(x_enc @ up.T), (n_facts, d_ff)."""
    with torch.no_grad():
        _, hidden = model.forward(model.facts["inputs"])
    return hidden.detach().cpu().float().numpy()


def _panel(ax, W, title, is_up, input_vocab):
    """Draw one weight matrix as a symmetric diverging heatmap (red = negative)."""
    vmax = float(np.abs(W).max()) or 1.0
    im = ax.imshow(W, aspect="auto", cmap=CMAP, vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=9)
    if is_up:
        # up_matrix columns are [one_hot(token0) | one_hot(token1)]; mark the split.
        ax.axvline(input_vocab - 0.5, color="black", lw=1.0)
        ax.set_xlabel("input one-hot  (token0 | token1)")
        ax.set_ylabel("hidden neuron")
    else:
        ax.set_xlabel("hidden neuron")
        ax.set_ylabel("output label")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def make_figure(d, thr, aam):
    """Build/train the three networks for one criterion and save a 3x2 figure."""
    input_vocab = 2 * d
    tag = f"acc{thr}_{aam}"
    print(f"\n===== criterion: accuracy_threshold={thr}, any_all_most={aam} =====")

    hc = _latest(TOPFRAC_LOG, d, thr, aam)
    hc_max, S, tf = hc["max_facts"], hc["best_S"], hc["best_top_fraction"]
    tr_max = _latest(FULLTRAIN_LOG, d, thr, aam)["max_facts"]
    print(f"  hand-coded: max_facts={hc_max}, S={S}, top_fraction={tf}")
    print(f"  trained:    max_facts={tr_max}")

    print("  building hand-coded (x11) ...")
    hc_model, hc_acc, hc_note = build_handcoded(d, hc_max, S, tf, aam)
    print(f"  -> hand-coded bga={hc_acc:.4f} ({hc_note})")

    print("  training trained @ trained-max (x11) ...")
    trmax_model, trmax_acc, trmax_note = train_trained(d, tr_max, aam)
    print(f"  -> trained@{tr_max} bga={trmax_acc:.4f} ({trmax_note})")

    print("  training trained @ hand-coded-max (x11) ...")
    trhc_model, trhc_acc, trhc_note = train_trained(d, hc_max, aam)
    print(f"  -> trained@{hc_max} bga={trhc_acc:.4f} ({trhc_note})")

    # (model, multi-line weight-panel label, short activation label, n_facts)
    rows = [
        (hc_model,
         f"hand-coded  (n_facts={hc_max}, S={S}, tf={tf})\nacc={hc_acc:.3f}  [{hc_note}]",
         f"hand-coded  (n_facts={hc_max})\nacc={hc_acc:.3f}  [{hc_note}]", hc_max),
        (trmax_model,
         f"trained @ trained-max  (n_facts={tr_max})\nacc={trmax_acc:.3f}  [{trmax_note}]",
         f"trained @ trained-max  (n_facts={tr_max})\nacc={trmax_acc:.3f}  [{trmax_note}]", tr_max),
        (trhc_model,
         f"trained @ hand-coded-max  (n_facts={hc_max})\nacc={trhc_acc:.3f}  [{trhc_note}]",
         f"trained @ hand-coded-max  (n_facts={hc_max})\nacc={trhc_acc:.3f}  [{trhc_note}]", hc_max),
    ]
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Weights figure: 3 rows (models) x 2 cols (mlp_in, mlp_out) ──
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    for i, (model, label, _, _) in enumerate(rows):
        up = _mat(model, "up_matrix")      # mlp_in  (d_ff, 2*input_vocab)
        down = _mat(model, "down_matrix")  # mlp_out (n_labels, d_ff)
        _panel(axes[i][0], up, f"{label}\nmlp_in (up_matrix) {up.shape}",
               is_up=True, input_vocab=input_vocab)
        _panel(axes[i][1], down, f"mlp_out (down_matrix) {down.shape}",
               is_up=False, input_vocab=input_vocab)
    fig.suptitle(
        f"d={d} weight matrices — criterion: accuracy≥{thr}, rule='{aam}'\n"
        f"rows: hand-coded / trained@trained-max / trained@hand-coded-max   "
        f"(cols: mlp_in, mlp_out)   |   red = negative",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    w_path = os.path.join(OUT_DIR, f"weights_d{d}_{tag}.png")
    fig.savefig(w_path, dpi=130)
    print(f"  saved {w_path}")

    # ── Activations figure: 1 row x 3 cols (models); each is (n_facts x d_ff) ──
    figa, axesa = plt.subplots(1, 3, figsize=(15, 8))
    for ax, (model, _, act_label, n_facts) in zip(axesa, rows):
        A = _activations(model)  # (n_facts, d_ff)
        targets = model.facts["targets"].detach().cpu().numpy()
        # Sort facts by label so each label's facts form a contiguous block
        # (generate_facts already does this, but be defensive).
        order = np.argsort(targets, kind="stable")
        A, targets = A[order], targets[order]
        vmax = float(np.abs(A).max()) or 1.0
        # interpolation="nearest" -> raw pixels, no blurring on the tall panels.
        im = ax.imshow(A, aspect="auto", cmap=CMAP, vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
        # Horizontal separators between groups of facts with different labels.
        for b in np.where(np.diff(targets) != 0)[0]:
            ax.axhline(b + 0.5, color="black", lw=0.6)
        ax.set_title(f"{act_label}\nhidden activations {A.shape}  "
                     f"(rows grouped by label)", fontsize=9)
        ax.set_xlabel("hidden neuron")
        ax.set_ylabel("fact  (sorted by label)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    figa.suptitle(
        f"d={d} neuron activations (relu hidden, per fact x neuron) — "
        f"criterion: accuracy≥{thr}, rule='{aam}'\n"
        f"post-ReLU so all values ≥ 0 (white→blue); red = negative",
        fontsize=12)
    figa.tight_layout(rect=[0, 0, 1, 0.93])
    a_path = os.path.join(OUT_DIR, f"activations_d{d}_{tag}.png")
    figa.savefig(a_path, dpi=130)
    print(f"  saved {a_path}")

    plt.show()
    return w_path, a_path


#%%
if __name__ == "__main__":
    paths = [make_figure(D, thr, aam) for thr, aam in CRITERIA]
    print("\nAll figures written:")
    for w_path, a_path in paths:
        print(" ", w_path)
        print(" ", a_path)
# %%
