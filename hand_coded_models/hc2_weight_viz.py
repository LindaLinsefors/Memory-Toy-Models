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
then keep the BEST attempt (highest final accuracy) — for all criteria,
including 'most'. The any/all/most rule only selects which capacity-log row
(n_facts, S, top_fraction) we build at, not which of the 11 nets we show. Trained
nets use NO early stopping — they run the full n_epochs and we visualise the
final-epoch weights.

Both the hand-coded and the trained nets are created once and saved to
weight_viz/model_cache/ (the whole model: up/down weights + facts); later runs load them
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
import matplotlib
matplotlib.use("Agg")  # save figures to files only; no interactive popup windows
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Bump every default text size so the saved figures are easy to read.
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.titlesize": 20,
})

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
    """Pick the attempt to display/save: always the BEST (highest accuracy)
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


# Save the whole hand-coded net (up/down + facts) so re-runs load the ENTIRE
# model straight from disk — no connection matrix, no simulated annealing, no
# reconstruction. Keyed by the params that fully determine it: (d, n_facts, S, tf).
_hc_cache = {}


def _hc_cache_path(d, n_facts, S, tf):
    return os.path.join(CACHE_DIR, f"handcoded_d{d}_nfacts{n_facts}_S{S}_tf{tf}.pt")


def _load_handcoded(d, n_facts, S, tf, inputs, targets, up, down):
    """Reconstruct a HandCodedModel2 purely from saved tensors — bypass __init__
    (so no conn matrix / annealing runs) and set only what forward()/evaluate()/the
    plots read: settings, facts, and the two weight tensors."""
    model = HandCodedModel2.__new__(HandCodedModel2)
    model.settings = _hc_settings(d, n_facts, S, tf)
    model.facts = {"inputs": inputs.to(DEV), "targets": targets.to(DEV)}
    model.up_matrix = up.to(DEV)
    model.down_matrix = down.to(DEV)
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
                                 r["up_matrix"], r["down_matrix"]),
                 r["accuracy"])
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
                accuracy, _, _ = model.evaluate()
                attempts.append((model, accuracy))
                records.append({
                    "attempt": a, "accuracy": accuracy,
                    "up_matrix": model.up_matrix.detach().cpu(),
                    "down_matrix": model.down_matrix.detach().cpu(),
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


def _rebuild_trained(d, n_facts, init_seed, up, down):
    """Reconstruct a FullyTrainedModel2 (facts are deterministic from seed/n_facts)
    and load saved weights into it, so it forwards/plots exactly as trained."""
    model = FullyTrainedModel2(
        input_vocab_size=2 * d, output_vocab_size=d, d_ff=d,
        n_facts=n_facts, seed=42, init_seed=init_seed)
    model.up_matrix = up.to(DEV)
    model.down_matrix = down.to(DEV)
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
                                  r["up_matrix"], r["down_matrix"]),
                 r["accuracy"])
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
                accuracy, _, _ = model.evaluate()
                attempts.append((model, accuracy))
                records.append({
                    "init_seed": a, "accuracy": accuracy,
                    "up_matrix": model.up_matrix.detach().cpu(),
                    "down_matrix": model.down_matrix.detach().cpu(),
                })
                print(f"      trained n_facts={n_facts} attempt {a}: accuracy={accuracy:.4f}")
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
    ax.set_title(title, fontsize=14)
    if is_up:
        # up_matrix columns are [one_hot(token0) | one_hot(token1)]; mark the split.
        ax.axvline(input_vocab - 0.5, color="black", lw=1.0)
        # Restart the tick numbering at the split so each token block reads 0..N-1.
        step = max(1, input_vocab // 4)
        ticks = list(range(0, 2 * input_vocab, step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([t % input_vocab for t in ticks])
        ax.set_xlabel("input, 2x one-hot  (token0 | token1)")
        ax.set_ylabel("hidden neuron")
    else:
        ax.set_xlabel("hidden neuron")
        ax.set_ylabel("output label")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _activation_figure(rows, d, tag, value_fn, fname, heading, signed=True):
    """Save a 1x3 (per-model) heatmap of a per-fact x neuron quantity.

    value_fn(model) -> (n_facts, d_ff). Facts are sorted by label with a black
    separator between label groups (same treatment for the post- and pre-ReLU
    versions). fname names the PNG (activations / preactivations); heading is the
    figure suptitle.

    signed=True (pre-ReLU): symmetric diverging scale -max..max (red/white/blue).
    signed=False (post-ReLU, all >= 0): sequential 0..max, white -> blue, no red.
    Returns the saved path."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    for ax, (model, _, act_title, n_facts) in zip(axes, rows):
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
        ax.set_title(act_title, fontsize=14)
        ax.set_xlabel("hidden neuron")
        ax.set_ylabel("fact  (sorted by label)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(heading, fontsize=20, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    path = os.path.join(OUT_DIR, f"{fname}_d{d}_{tag}.png")
    fig.savefig(path, dpi=130)
    print(f"  saved {path}")
    return path


def _histogram_figure(rows, d, tag):
    """Save a 3x2 grid of weight-value histograms: one row per model, columns
    (mlp_in / up_matrix, mlp_out / down_matrix). Each panel is a histogram over all
    entries of that weight matrix. Returns the saved path."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    for i, (model, weight_title, _, _) in enumerate(rows):
        up = _mat(model, "up_matrix").ravel()      # mlp_in
        down = _mat(model, "down_matrix").ravel()  # mlp_out
        for ax, W, title in ((axes[i][0], up, weight_title("embedding matrices")),
                             (axes[i][1], down, weight_title("unembedding matrix"))):
            ax.hist(W, bins=60, color="steelblue", edgecolor="none")
            ax.axvline(0.0, color="black", lw=0.8)  # mark zero for reference
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("weight value")
            ax.set_ylabel("count")
    fig.suptitle("Weight Histograms", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    path = os.path.join(OUT_DIR, f"weight_hist_d{d}_{tag}.png")
    fig.savefig(path, dpi=130)
    print(f"  saved {path}")
    return path


def make_figure(d, thr, aam):
    """Build/train the three networks for one criterion and save a 3x2 figure.
    Reads the (n_facts, S, top_fraction) params from the capacity logs, then
    delegates to _render_figure."""
    print(f"\n===== criterion: accuracy_threshold={thr}, any_all_most={aam} =====")
    hc = _latest(TOPFRAC_LOG, d, thr, aam)
    hc_max, S, tf = hc["max_facts"], hc["best_S"], hc["best_top_fraction"]
    tr_max = _latest(FULLTRAIN_LOG, d, thr, aam)["max_facts"]
    print(f"  hand-coded: max_facts={hc_max}, S={S}, top_fraction={tf}")
    print(f"  trained:    max_facts={tr_max}")
    return _render_figure(d, f"acc{thr}_{aam}", hc_max, S, tf, tr_max, aam)


def _render_figure(d, tag, hc_max, S, tf, tr_max, aam):
    """Build the hand-coded net (11 attempts, best kept) at (hc_max, S, tf) and the
    two trained nets (at hc_max and tr_max, reused from cache when present), then save
    the 3x2 weight figure plus the two activation figures. Returns their paths."""
    input_vocab = 2 * d

    print("  building hand-coded (x11) ...")
    hc_model, hc_acc, hc_note = build_handcoded(d, hc_max, S, tf, aam)
    print(f"  -> hand-coded accuracy={hc_acc:.4f} ({hc_note})")

    print("  training trained @ trained-max (x11) ...")
    trmax_model, trmax_acc, trmax_note = train_trained(d, tr_max, aam)
    print(f"  -> trained@{tr_max} accuracy={trmax_acc:.4f} ({trmax_note})")

    print("  training trained @ hand-coded-max (x11) ...")
    trhc_model, trhc_acc, trhc_note = train_trained(d, hc_max, aam)
    print(f"  -> trained@{hc_max} accuracy={trhc_acc:.4f} ({trhc_note})")

    # (model, weight-title builder(type_str)->str, activation-panel title, n_facts)
    # Order: hand-coded, then trained at the SAME n_facts (middle), then trained at
    # its own max. The hand-coded titles list its hyper-parameters (S, top_fraction).
    hc_act = (f"hand-coded \n n_facts={hc_max}, acc={hc_acc:.3f}, "
              f"S={S} top_fraction={tf}")

    def hc_weight(type):
        return (f"hand-coded {type} \n n_facts={hc_max}, acc={hc_acc:.3f}, "
                f"S={S} top_fraction={tf}")

    trhc_act = f"trained \n n_facts={hc_max}, acc={trhc_acc:.3f}"

    def trhc_weight(type):
        return f"trained {type} \n n_facts={hc_max}, acc={trhc_acc:.3f}"

    trmax_act = f"trained \n n_facts={tr_max}, acc={trmax_acc:.3f}"

    def trmax_weight(type):
        return f"trained {type} \n n_facts={tr_max}, acc={trmax_acc:.3f}"

    rows = [
        (hc_model, hc_weight, hc_act, hc_max),
        (trhc_model, trhc_weight, trhc_act, hc_max),
        (trmax_model, trmax_weight, trmax_act, tr_max),
    ]
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Weights figure: 3 rows (models) x 2 cols (mlp_in, mlp_out) ──
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    for i, (model, weight_title, _, _) in enumerate(rows):
        up = _mat(model, "up_matrix")      # mlp_in  (d_ff, 2*input_vocab)
        down = _mat(model, "down_matrix")  # mlp_out (n_labels, d_ff)
        _panel(axes[i][0], up, weight_title("embedding matrices"),
               is_up=True, input_vocab=input_vocab)
        _panel(axes[i][1], down, weight_title("unembedding matrix"),
               is_up=False, input_vocab=input_vocab)
    fig.suptitle("Weight Matrices", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    w_path = os.path.join(OUT_DIR, f"weights_d{d}_{tag}.png")
    fig.savefig(w_path, dpi=130)
    print(f"  saved {w_path}")

    # ── Activation figures: post-ReLU, and a second "pre-activation" version ──
    a_path = _activation_figure(
        rows, d, tag, _activations,
        fname="activations", heading="Neuron Activations",
        signed=False)
    pre_path = _activation_figure(
        rows, d, tag, _preactivations,
        fname="preactivations", heading="Neuron Pre-Activations",
        signed=True)

    # ── Weight-histogram figure: same 3 rows x 2 cols layout as the heatmaps ──
    hist_path = _histogram_figure(rows, d, tag)

    plt.close("all")  # free the figures; nothing is shown (Agg backend, files only)
    return w_path, a_path, pre_path, hist_path


def make_handcoded_S4_tf01_figure(d):
    """Like the acc1.0_any figure but the hand-coded net uses S=4, top_fraction=0.1.
    Hand-coded and one trained net share n_facts=64; a second trained net is shown at
    n_facts=568. Both trained nets are reused from cache (no retraining)."""
    print("\n===== hand-coded S=4, top_fraction=0.1 (n_facts=64; trained @ 64, 568) =====")
    return _render_figure(d, "handcoded_S4_tf0.1", hc_max=64, S=4, tf=0.1, tr_max=568,
                          aam="any")


#%%
if __name__ == "__main__":
    paths = [make_figure(D, thr, aam) for thr, aam in CRITERIA]
    paths.append(make_handcoded_S4_tf01_figure(D))
    print("\nAll figures written:")
    for w_path, a_path, pre_path, hist_path in paths:
        print(" ", w_path)
        print(" ", a_path)
        print(" ", pre_path)
        print(" ", hist_path)
# %%
