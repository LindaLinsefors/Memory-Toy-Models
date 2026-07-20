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
then keep the BEST attempt (highest final best_guess_accuracy) — for all criteria,
including 'most'. The any/all/most rule only selects which capacity-log row
(n_facts, S, top_fraction) we build at, not which of the 11 nets we show. Trained
nets use NO early stopping — they run the full n_epochs and we visualise the
final-epoch weights.

Both the hand-coded and the trained nets are created once and saved to
weight_viz/model_cache/ (the whole model: weights + facts); later runs load them
straight from there instead of rebuilding/retraining. Only scalar hyper-parameters
come from the logs. Figures are written to hand_coded_models/weight_viz/.
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
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import models  # noqa: F401  -- importing sets the default torch device (cuda/cpu)
from hc2 import HandCodedModel2, HandCodedModel2Settings, get_conn_matrix
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
    """Pick the attempt to display/save: always the BEST (highest best_guess_accuracy)
    of the N attempts, regardless of the criterion's any/all/most rule.

    The any/all/most rule only selects which capacity-log row (n_facts, S,
    top_fraction) we build at; the net we keep is always the best of the 11 — e.g.
    for the 'most' criterion we still show the best run, not the median one.
    `aam` is kept in the signature (callers thread it through) but no longer affects
    the choice. Returns (model, accuracy, rank_note)."""
    model, acc = max(models_and_accs, key=lambda ma: ma[1])
    return model, acc, f"best of {len(models_and_accs)}"


# ── Model builders (11 attempts each) ─────────────────────────────────────────

def _hc_settings(d, n_facts, S, top_fraction):
    return HandCodedModel2Settings(
        input_vocab_size=2 * d,
        output_vocab_size=d,
        d_ff=d,
        n_facts=n_facts,
        n_neurons_per_label=S,
        use_top_n_or_top_fraction="top_fraction",
        top_fraction=top_fraction,
        seed=42,
    )


# Save the whole hand-coded net (up/down/bias + facts) so re-runs load the ENTIRE
# model straight from disk — no connection matrix, no simulated annealing, no
# reconstruction. Keyed by the params that fully determine it: (d, n_facts, S, tf).
_hc_cache = {}


def _hc_cache_path(d, n_facts, S, tf):
    return os.path.join(CACHE_DIR, f"handcoded_d{d}_nfacts{n_facts}_S{S}_tf{tf}.pt")


def _load_handcoded(d, n_facts, S, tf, inputs, targets, up, down, bias):
    """Reconstruct a HandCodedModel2 purely from saved tensors — bypass __init__
    (so no conn matrix / annealing runs) and set only what forward()/evaluate()/the
    plots read: settings, facts, and the three weight tensors."""
    model = HandCodedModel2.__new__(HandCodedModel2)
    model.settings = _hc_settings(d, n_facts, S, tf)
    model.facts = {"inputs": inputs.to(DEV), "targets": targets.to(DEV)}
    model.up_matrix = up.to(DEV)
    model.down_matrix = down.to(DEV)
    model.down_bias = bias.to(DEV)
    return model


def build_handcoded(d, n_facts, S, top_fraction, aam):
    """Return the criterion's representative among N_ATTEMPTS HandCodedModel2 nets.

    Loads the whole 11-net bundle from disk if cached (the entire model — no conn
    matrix needed); otherwise builds them once — the conn matrix is built a single
    time and shared across attempts (they differ only via the constructor's randperm
    tie-breaking) — and saves the full nets for reuse."""
    key = (d, n_facts, S, top_fraction)
    if key not in _hc_cache:
        path = _hc_cache_path(d, n_facts, S, top_fraction)
        if os.path.exists(path):
            blob = torch.load(path, map_location="cpu")
            attempts = [
                (_load_handcoded(d, n_facts, S, top_fraction,
                                 blob["inputs"], blob["targets"],
                                 r["up_matrix"], r["down_matrix"], r["down_bias"]),
                 r["bga"])
                for r in blob["attempts"]
            ]
            print(f"      hand-coded n_facts={n_facts}: loaded {len(attempts)} "
                  f"attempts from cache")
        else:
            conn = get_conn_matrix(d, d, S, 42)  # built once, shared by all attempts
            attempts, records = [], []
            for a in range(N_ATTEMPTS):
                # Seed the global RNG so the constructor's randperm tie-breaking
                # gives 11 distinct, reproducible builds.
                torch.manual_seed(a)
                model = HandCodedModel2(_hc_settings(d, n_facts, S, top_fraction),
                                        precomputed_conn=conn)
                _, bga, _, _ = model.evaluate()
                attempts.append((model, bga))
                records.append({
                    "attempt": a, "bga": bga,
                    "up_matrix": model.up_matrix.detach().cpu(),
                    "down_matrix": model.down_matrix.detach().cpu(),
                    "down_bias": model.down_bias.detach().cpu(),
                })
            os.makedirs(CACHE_DIR, exist_ok=True)
            facts0 = attempts[0][0].facts
            torch.save({"d": d, "n_facts": n_facts, "S": S,
                        "top_fraction": top_fraction,
                        "inputs": facts0["inputs"].detach().cpu(),
                        "targets": facts0["targets"].detach().cpu(),
                        "attempts": records}, path)
            print(f"      saved {N_ATTEMPTS} hand-coded nets -> {path}")
        _hc_cache[key] = attempts
    return _select(_hc_cache[key], aam)


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

# Diverging colormap with RED = NEGATIVE, WHITE = 0, blue = positive.
# bwr_r = red->white->blue; unlike coolwarm its midpoint is pure white, so a
# zero value (the center of the symmetric vmin/vmax scale) renders white, not gray.
CMAP = "bwr_r"

# Sequential white->blue for non-negative data (post-ReLU activations): scaled
# 0 (white) -> max (blue), no red half. Matches the positive end of bwr_r so blue
# means the same thing in every figure.
WHITE_BLUE = LinearSegmentedColormap.from_list("white_blue", ["white", (0.0, 0.0, 1.0)])


def _mat(model, name):
    return getattr(model, name).detach().cpu().float().numpy()


def _activations(model):
    """Hidden-layer activations for every fact: relu(x_enc @ up.T), (n_facts, d_ff)."""
    with torch.no_grad():
        _, hidden = model.forward(model.facts["inputs"])
    return hidden.detach().cpu().float().numpy()


def _preactivations(model):
    """Pre-activations (before ReLU): x_enc @ up_matrix.T, (n_facts, d_ff).

    forward() only returns the post-ReLU hidden, so recompute the same one-hot
    input encoding here and stop before the relu. Works for both model classes
    (both expose up_matrix, settings.input_vocab_size and facts['inputs'])."""
    x = model.facts["inputs"]
    iv = model.settings.input_vocab_size
    with torch.no_grad():
        first = F.one_hot(x[:, 0], num_classes=iv).float()
        second = F.one_hot(x[:, 1], num_classes=iv).float()
        x_enc = torch.cat([first, second], dim=-1)
        pre = x_enc @ model.up_matrix.T
    return pre.detach().cpu().float().numpy()


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


def _activation_figure(rows, d, thr, aam, tag, value_fn, fname, panel_word,
                       heading, note, signed=True):
    """Save a 1x3 (per-model) heatmap of a per-fact x neuron quantity.

    value_fn(model) -> (n_facts, d_ff). Facts are sorted by label with a black
    separator between label groups (same treatment for the post- and pre-ReLU
    versions). fname names the PNG (activations / preactivations); panel_word is
    the per-axis phrase and heading / note are the two suptitle lines.

    signed=True (pre-ReLU): symmetric diverging scale -max..max (red/white/blue).
    signed=False (post-ReLU, all >= 0): sequential 0..max, white -> blue, no red.
    Returns the saved path."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    for ax, (model, _, act_label, n_facts) in zip(axes, rows):
        A = value_fn(model)  # (n_facts, d_ff)
        targets = model.facts["targets"].detach().cpu().numpy()
        # Sort facts by label so each label's facts form a contiguous block.
        order = np.argsort(targets, kind="stable")
        A, targets = A[order], targets[order]
        if signed:
            vmax = float(np.abs(A).max()) or 1.0
            vmin, cmap = -vmax, CMAP
        else:  # non-negative data: 0 (white) -> max (blue)
            vmax = float(A.max()) or 1.0
            vmin, cmap = 0.0, WHITE_BLUE
        # interpolation="nearest" -> raw pixels, no blurring on the tall panels.
        im = ax.imshow(A, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        # Horizontal separators between groups of facts with different labels.
        for b in np.where(np.diff(targets) != 0)[0]:
            ax.axhline(b + 0.5, color="black", lw=0.6)
        ax.set_title(f"{act_label}\n{panel_word} {A.shape}  "
                     f"(rows grouped by label)", fontsize=9)
        ax.set_xlabel("hidden neuron")
        ax.set_ylabel("fact  (sorted by label)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"d={d} {heading} — criterion: accuracy≥{thr}, rule='{aam}'\n{note}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUT_DIR, f"{fname}_d{d}_{tag}.png")
    fig.savefig(path, dpi=130)
    print(f"  saved {path}")
    return path


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

    # (model, multi-line weight-panel label, activation-panel label, n_facts)
    # Order: hand-coded, then trained at the SAME n_facts (middle), then trained at
    # its own max. The hand-coded labels list its hyper-parameters (S, top_fraction).
    hc_label = (f"hand-coded  (n_facts={hc_max}, S={S}, top_fraction={tf})\n"
                f"acc={hc_acc:.3f}  [{hc_note}]")
    rows = [
        (hc_model, hc_label, hc_label, hc_max),
        (trhc_model,
         f"trained @ hand-coded-max  (n_facts={hc_max})\nacc={trhc_acc:.3f}  [{trhc_note}]",
         f"trained @ hand-coded-max  (n_facts={hc_max})\nacc={trhc_acc:.3f}  [{trhc_note}]", hc_max),
        (trmax_model,
         f"trained @ trained-max  (n_facts={tr_max})\nacc={trmax_acc:.3f}  [{trmax_note}]",
         f"trained @ trained-max  (n_facts={tr_max})\nacc={trmax_acc:.3f}  [{trmax_note}]", tr_max),
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
        f"rows: hand-coded / trained@hand-coded-max / trained@trained-max   "
        f"(cols: mlp_in, mlp_out)   |   red = negative",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    w_path = os.path.join(OUT_DIR, f"weights_d{d}_{tag}.png")
    fig.savefig(w_path, dpi=130)
    print(f"  saved {w_path}")

    # ── Activation figures: post-ReLU, and a second "pre-activation" version ──
    a_path = _activation_figure(
        rows, d, thr, aam, tag, _activations,
        fname="activations", panel_word="hidden activations",
        heading="neuron activations (relu hidden, per fact x neuron)",
        note="post-ReLU (all values ≥ 0); scale 0 (white) → max (blue)",
        signed=False)
    pre_path = _activation_figure(
        rows, d, thr, aam, tag, _preactivations,
        fname="preactivations", panel_word="hidden pre-activations",
        heading="neuron pre-activations (x_enc @ up.T, before ReLU)",
        note="pre-ReLU: 0 = white, positive = blue, negative = red",
        signed=True)

    plt.show()
    return w_path, a_path, pre_path


#%%
if __name__ == "__main__":
    paths = [make_figure(D, thr, aam) for thr, aam in CRITERIA]
    print("\nAll figures written:")
    for w_path, a_path, pre_path in paths:
        print(" ", w_path)
        print(" ", a_path)
        print(" ", pre_path)
# %%
