#%%
"""Plots for the HandCodedModel2 top_fraction x S sweep.

Reads the JSON result files written by hc2_sweep.py (under
hand_coded_models/hc2_sweep_results/) and visualises best_guess_accuracy as a
function of top_fraction and n_neurons_per_label (S).
"""

import os
import json
import glob
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def _find_results_dir():
    """Locate hand_coded_models/hc2_sweep_results robustly.

    When run as a script, __file__ points here and the sibling folder is obvious.
    When run interactively (#%% cells / Jupyter), __file__ may be undefined or
    resolve against a different cwd, so we also walk up from __file__ and the cwd
    looking for the results folder.
    """
    candidates = []
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(here, "hc2_sweep_results"))
    except NameError:
        here = None

    # Walk upward from likely starting points, checking for the folder directly
    # or nested under hand_coded_models/.
    starts = [p for p in (here, os.getcwd()) if p]
    for start in starts:
        cur = start
        while True:
            candidates.append(os.path.join(cur, "hc2_sweep_results"))
            candidates.append(os.path.join(cur, "hand_coded_models", "hc2_sweep_results"))
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent

    for c in candidates:
        if os.path.isdir(c):
            return c
    # Nothing found: fall back to the sibling-of-this-file guess for a clear error.
    return candidates[0] if candidates else "hc2_sweep_results"


RESULTS_DIR = _find_results_dir()


def _load_records(d, n_facts, results_dir=RESULTS_DIR, include_test=False):
    """Load and concatenate all sweep records matching (d, n_facts).

    Scans every JSON file in `results_dir`, keeps those whose settings match the
    requested d and n_facts, and pools their `results` lists together (so several
    runs of the same setting are combined into more attempts). Files whose name
    starts with "test_" are skipped unless include_test=True.

    Returns (records, settings) where records is a list of dicts and settings is
    the settings block of the first matching file.
    """
    records = []
    settings = None
    # top_fraction grid files live in the top_fraction_grids/ subfolder (see
    # hc2_sweep_results/README.md). Match the grid name pattern there; optionally
    # also the test_* ones.
    tf_dir = os.path.join(results_dir, "top_fraction_grids")
    paths = glob.glob(os.path.join(tf_dir, "hc2_sweep_d*_nfacts*.json"))
    if include_test:
        paths += glob.glob(os.path.join(tf_dir, "test_hc2_sweep_d*_nfacts*.json"))
    for path in sorted(paths):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        s = payload.get("settings", {})
        if s.get("d") != d or s.get("n_facts") != n_facts:
            continue
        if settings is None:
            settings = s
        records.extend(payload["results"])

    if not records:
        raise FileNotFoundError(
            f"No sweep results for d={d}, n_facts={n_facts} in {results_dir}"
        )
    return records, settings


def _aggregate(records):
    """Aggregate records into per-(S, top_fraction) grids.

    Returns a dict with sorted axis values and 2-D arrays (rows = S, cols =
    top_fraction) of the mean, variance, count and max of best_guess_accuracy.
    """
    S_values = sorted({r["S"] for r in records})
    tf_values = sorted({r["top_fraction"] for r in records})
    s_index = {s: i for i, s in enumerate(S_values)}
    tf_index = {tf: j for j, tf in enumerate(tf_values)}

    # Collect every attempt's accuracy per cell, then reduce.
    buckets = {(i, j): [] for i in range(len(S_values)) for j in range(len(tf_values))}
    for r in records:
        buckets[(s_index[r["S"]], tf_index[r["top_fraction"]])].append(
            r["best_guess_accuracy"]
        )

    shape = (len(S_values), len(tf_values))
    mean = np.full(shape, np.nan)
    var = np.full(shape, np.nan)
    count = np.zeros(shape, dtype=int)
    best = np.full(shape, np.nan)
    for (i, j), vals in buckets.items():
        if vals:
            arr = np.asarray(vals)
            mean[i, j] = arr.mean()
            var[i, j] = arr.var()
            best[i, j] = arr.max()
            count[i, j] = arr.size

    return {
        "S_values": S_values,
        "tf_values": tf_values,
        "mean": mean,
        "var": var,
        "best": best,
        "count": count,
    }


def show_grid(d, n_facts):
    """
    Finds all data for the given d and n_facts in hc2_sweep_results and
    plots a grid where the y axis is S and the x axis is top_fraction,
    with the color of each cell representing the average best_guess_accuracy
    for all attempts for that (S, top_fraction) pair, and the correct d and n_facts.

    In each cell, write the average best_guess_accuracy and
    the variance of best_guess_accuracy across all attempts for that (S, top_fraction) pair.
    """
    records, _ = _load_records(d, n_facts)
    agg = _aggregate(records)
    S_values, tf_values = agg["S_values"], agg["tf_values"]
    mean, var, best = agg["mean"], agg["var"], agg["best"]
    n_attempts = int(agg["count"].max())

    fig, ax = plt.subplots(figsize=(1.1 * len(tf_values) + 2, 0.8 * len(S_values) + 2))
    im = ax.imshow(mean, origin="lower", aspect="auto", cmap="viridis",
                   vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(tf_values)))
    ax.set_xticklabels([f"{tf:.1f}" for tf in tf_values])
    ax.set_yticks(range(len(S_values)))
    ax.set_yticklabels(S_values)
    ax.set_xlabel("top_fraction")
    ax.set_ylabel("S (n_neurons_per_label)")

    # Annotate each cell with mean and variance; pick a legible text colour.
    for i in range(len(S_values)):
        for j in range(len(tf_values)):
            if np.isnan(mean[i, j]):
                continue
            text_color = "white" if mean[i, j] < 0.6 else "black"
            ax.text(j, i, f"{mean[i, j]:.3f}\n±{var[i, j]:.6f}",
                    ha="center", va="center", color=text_color, fontsize=7)

    # Highlight every cell that had at least one perfect (100%) run, in lime.
    n_perfect = 0
    for i in range(len(S_values)):
        for j in range(len(tf_values)):
            if best[i, j] >= 1.0:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor="lime", linewidth=2.0))
                n_perfect += 1

    # Highlight the cell with the highest mean accuracy, in red (drawn on top).
    best_i, best_j = np.unravel_index(np.nanargmax(mean), mean.shape)
    ax.add_patch(plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                               fill=False, edgecolor="red", linewidth=2.5))

    # Legend explaining the two highlight colours.
    handles = [
        plt.Line2D([0], [0], color="red", lw=2.5, label="highest mean accuracy"),
        plt.Line2D([0], [0], color="lime", lw=2.0,
                   label="≥ 1 run at 100% accuracy"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.18, 1.0),
              fontsize=8, framealpha=0.9)

    ax.set_title(
        f"HandCodedModel2 best_guess_accuracy\n"
        f"d={d}, n_facts={n_facts}, attempts={n_attempts}  |  "
        f"best: S={S_values[best_i]}, top_fraction={tf_values[best_j]:.1f} "
        f"({mean[best_i, best_j]:.3f})  |  {n_perfect} cell(s) with a 100% run"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean best_guess_accuracy")
    fig.tight_layout()
    plt.show()
    return fig


def plot_accuracy_vs_top_fraction(d, n_facts):
    """Line plot: mean best_guess_accuracy vs top_fraction, one line per S.

    Shaded band shows ± one standard deviation across attempts. Makes the optimal
    top_fraction for each S easy to read off.
    """
    records, _ = _load_records(d, n_facts)
    agg = _aggregate(records)
    S_values, tf_values = agg["S_values"], agg["tf_values"]
    mean, var = agg["mean"], agg["var"]
    std = np.sqrt(var)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(S_values)))
    x = np.asarray(tf_values)
    for i, S in enumerate(S_values):
        ax.plot(x, mean[i], "-o", color=colors[i], label=f"S={S}", markersize=4)
        ax.fill_between(x, mean[i] - std[i], mean[i] + std[i],
                        color=colors[i], alpha=0.12)

    ax.set_xlabel("top_fraction")
    ax.set_ylabel("mean best_guess_accuracy")
    ax.set_title(f"Accuracy vs top_fraction (d={d}, n_facts={n_facts})")
    ax.set_ylim(0, 1.02)
    ax.grid(True, ls="--", linewidth=0.5)
    ax.legend(ncol=2, fontsize=8, title="S")
    fig.tight_layout()
    plt.show()
    return fig


def plot_best_vs_S(d, n_facts):
    """For each S, the best top_fraction's accuracy vs S.

    Two lines: the best *mean* accuracy (averaged over attempts) and the best
    *single attempt* — the latter is what you'd actually pick when constructing a
    model, since the sweep keeps the best random construction.
    """
    records, _ = _load_records(d, n_facts)
    agg = _aggregate(records)
    S_values, tf_values = agg["S_values"], agg["tf_values"]
    mean, best = agg["mean"], agg["best"]

    best_mean = np.nanmax(mean, axis=1)
    best_attempt = np.nanmax(best, axis=1)
    argmax_tf = np.nanargmax(mean, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(S_values, best_mean, "-o", label="best mean over attempts")
    ax.plot(S_values, best_attempt, "-s", label="best single attempt")
    for i, S in enumerate(S_values):
        ax.annotate(f"tf={tf_values[argmax_tf[i]]:.1f}",
                    (S, best_mean[i]), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=7)

    ax.set_xlabel("S (n_neurons_per_label)")
    ax.set_ylabel("best_guess_accuracy at best top_fraction")
    ax.set_title(f"Best accuracy vs S (d={d}, n_facts={n_facts})")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(S_values)
    ax.grid(True, ls="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    plt.show()
    return fig


def load_capacity_results(path=None):
    """Read the capacity-search results written by hc2_capacity_search.py.

    That file (hc2_sweep_results/capacity_search_results.json) is JSONL: one JSON
    object per line, appended one per run. Returns the list of run dicts (in file
    order). Returns [] if the file does not exist yet.
    """
    if path is None:
        path = os.path.join(RESULTS_DIR, "capacity_search_results.json")
    if not os.path.exists(path):
        return []
    runs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def write_sorted_capacity_results(path=None, out_path=None):
    """Read the capacity-search results, sort them, and write a sorted copy.

    The rows are identical to the input (nothing added or dropped); they are just
    reordered into a sensible, stable order: by model size d, then the success
    criterion (accuracy_threshold, any/all/most), then max_facts. Output is the
    same JSONL format (one run per line) as capacity_search_results_sorted.json.

    Returns the sorted list of run dicts.
    """
    if path is None:
        path = os.path.join(RESULTS_DIR, "capacity_search_results.json")
    if out_path is None:
        out_path = os.path.join(RESULTS_DIR, "capacity_search_results_sorted.json")

    runs = load_capacity_results(path)

    def sort_key(r):
        return (
            r.get("d", 0),
            r.get("accuracy_threshold", 0),
            str(r.get("any_all_most", "")),
            r.get("max_facts", 0),
        )

    runs_sorted = sorted(runs, key=sort_key)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in runs_sorted:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(runs_sorted)} sorted rows to {out_path}")
    return None


def plot_capacity_vs_d(path=None):
    """Log-log plot of max_facts vs d, one line per (accuracy_threshold, any_all_most).

    Color encodes any_all_most; line style encodes accuracy_threshold. Reads the
    capacity-search log (JSONL) via load_capacity_results. Points with max_facts<=0
    (e.g. a search that bottomed out) are dropped, since they can't appear on a log
    axis.

    Also fits one power law (max_facts = C * d^k, i.e. a straight line in log-log
    space) per accuracy_threshold, pooling the any/all/most points together, and
    draws each fit in black with the threshold's line style. The fitted formula is
    shown in the legend.
    """
    if path is None:
        path = os.path.join(RESULTS_DIR, "capacity_search_results.json")
    runs = load_capacity_results(path)
    if not runs:
        raise FileNotFoundError(f"No capacity results found in {path}")

    # color = any_all_most, line+marker style = accuracy_threshold.
    aam_color = {"any": "tab:blue", "all": "tab:red", "most": "tab:green"}
    aam_rank = {"any": 0, "most": 1, "all": 2}
    # Explicit styles for the thresholds we care about; fall back for any others.
    thr_style = {1.0: "-o", 0.9: "--x"}
    fallback_styles = ["-o", "--x", ":s", "-.^"]
    thresholds = sorted({r.get("accuracy_threshold") for r in runs})
    for i, t in enumerate(thresholds):
        thr_style.setdefault(t, fallback_styles[i % len(fallback_styles)])

    # Group rows by (accuracy_threshold, any_all_most) -> {d: max_facts}.
    groups = defaultdict(dict)
    for r in runs:
        groups[(r["accuracy_threshold"], r["any_all_most"])][r["d"]] = r["max_facts"]

    fig, ax = plt.subplots(figsize=(8, 6))
    all_ys = []
    # Legend order: lower accuracy_threshold first, then any/most/all.
    for (thr, aam) in sorted(groups, key=lambda k: (k[0], aam_rank.get(k[1], 99))):
        ds_mf = sorted(groups[(thr, aam)].items())
        xs = [d for d, mf in ds_mf if mf and mf > 0]
        ys = [mf for d, mf in ds_mf if mf and mf > 0]
        if not xs:
            continue
        all_ys.extend(ys)
        ax.loglog(xs, ys, thr_style[thr], markersize=5,
                  color=aam_color.get(aam, "tab:gray"),
                  label=f"{aam}, acc>={thr}")

    # Power-law fits (straight lines in log-log space), one per accuracy_threshold,
    # pooling the any/all/most points. y = C * d^k.
    thr_fit_ls = {1.0: "-", 0.9: "--"}
    fallback_ls = ["-", "--", ":", "-."]
    for i, t in enumerate(thresholds):
        thr_fit_ls.setdefault(t, fallback_ls[i % len(fallback_ls)])

    thr_points = defaultdict(list)
    for (thr, aam), dmap in groups.items():
        for dd, mf in dmap.items():
            if mf and mf > 0:
                thr_points[thr].append((dd, mf))

    for thr in sorted(thr_points):
        fx = np.array([d for d, _ in sorted(thr_points[thr])], dtype=float)
        fy = np.array([mf for _, mf in sorted(thr_points[thr])], dtype=float)
        if len(fx) < 2:
            continue
        k, b = np.polyfit(np.log(fx), np.log(fy), 1)
        C = np.exp(b)
        xline = np.array([fx.min(), fx.max()])
        ax.loglog(xline, C * xline ** k, color="black", alpha=0.5,
                  linestyle=thr_fit_ls[thr], linewidth=1.5,
                  label=f"fit acc>={thr}: {C:.3g}·d^{k:.2f}")

    ax.set_xlabel("d (model size)")
    ax.set_ylabel("max_facts")
    ax.set_title("Capacity vs d  (color = any/all/most, style = accuracy_threshold; "
                 "black = power-law fit)")

    # x ticks: the model sizes we sweep, as plain integers (no minor 10^x ticks).
    xticks = [16, 32, 64, 128, 256]
    ax.set_xticks(xticks)
    ax.set_xticks([], minor=True)
    ax.set_xticklabels([str(x) for x in xticks])

    # y ticks: powers of two spanning the plotted max_facts range.
    if all_ys:
        lo = int(np.floor(np.log2(min(all_ys))))
        hi = int(np.ceil(np.log2(max(all_ys))))
        yticks = [2 ** n for n in range(lo, hi + 1)]
        ax.set_yticks(yticks)
        ax.set_yticks([], minor=True)
        ax.set_yticklabels([str(y) for y in yticks])

    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
    return fig


'''
if __name__ == "__main__":
    d = 32
    n_facts = 256
    show_grid(d, n_facts)
    plot_accuracy_vs_top_fraction(d, n_facts)
    plot_best_vs_S(d, n_facts)
'''


#%%
d = 32
for n_facts in [32, 64, 128, 256, 512, 1024]:
    show_grid(d, n_facts)
# %%
d = 256
for n_facts in [512, 1024, 2048, 4096, 8192]:
    show_grid(d, n_facts)
# %%
plot_capacity_vs_d()
# %%
write_sorted_capacity_results()
# %%

_ = plot_capacity_vs_d(path=os.path.join(RESULTS_DIR, 
                                         "capacity_search_results.json"))
# %%
_ = plot_capacity_vs_d(path=os.path.join(RESULTS_DIR, 
                                         "capacity_search_results_topn.json"))

# %%
