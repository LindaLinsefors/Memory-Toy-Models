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
              fontsize=9, framealpha=0.9)

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
    ax.legend(ncol=2, fontsize=9, title="S")
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


def _draw_capacity_vs_d(ax, path=None, aam_color=None, decorate=True,
                        y_field="max_facts", y_log=True):
    """Draw the hc2 capacity-vs-d plot onto a given Axes (no figure / no show).

    Color encodes any_all_most; line style encodes accuracy_threshold. Reads the
    capacity-search log (JSONL) via load_capacity_results. Points with y_field<=0
    (e.g. a search that bottomed out) are dropped, since they can't appear on a log
    axis.

    y_field selects which column to plot on the y axis: one of "max_facts" (default),
    "best_top_n", "best_S" or "best_top_fraction".

    y_log toggles the y-axis scale: log (default) or linear. The x axis is always
    log. With a linear y axis the powers-of-two y ticks are dropped (matplotlib
    auto-ticks instead) and the power-law fits render as curves rather than lines.

    Also fits one power law (y_field = C * d^k, i.e. a straight line in log-log
    space) per accuracy_threshold, pooling the any/all/most points together, and
    draws each fit in black with the threshold's line style. The fitted formula is
    shown in the legend.

    aam_color overrides the any/most/all colors. decorate=False skips the axis
    labels/ticks/title/legend (so a caller can overlay several datasets and set
    those once). Returns the list of plotted y_field values (for shared tick ranges).
    """
    if path is None:
        path = os.path.join(RESULTS_DIR, "capacity_search_results.json")
    runs = load_capacity_results(path)
    if not runs:
        raise FileNotFoundError(f"No capacity results found in {path}")

    # x is always log; y toggles between log and linear.
    ax.set_xscale("log")
    ax.set_yscale("log" if y_log else "linear")

    # color = any_all_most, line+marker style = accuracy_threshold.
    if aam_color is None:
        aam_color = {"any": "tab:blue", "all": "tab:red", "most": "tab:green"}
    aam_rank = {"any": 0, "most": 1, "all": 2}
    # Explicit styles for the thresholds we care about; fall back for any others.
    thr_style = {1.0: "-o", 0.9: "--x"}
    fallback_styles = ["-o", "--x", ":s", "-.^"]
    thresholds = sorted({r.get("accuracy_threshold") for r in runs})
    for i, t in enumerate(thresholds):
        thr_style.setdefault(t, fallback_styles[i % len(fallback_styles)])

    # Group rows by (accuracy_threshold, any_all_most) -> {d: y_field}.
    groups = defaultdict(dict)
    for r in runs:
        groups[(r["accuracy_threshold"], r["any_all_most"])][r["d"]] = r[y_field]

    all_ys = []
    # On a log y axis, zeros (or negatives) can't be shown, so drop them; on a
    # linear axis keep them (0 is a meaningful position there).
    keep = (lambda mf: mf is not None and mf > 0) if y_log \
        else (lambda mf: mf is not None)
    # Legend order: lower accuracy_threshold first, then any/most/all.
    for (thr, aam) in sorted(groups, key=lambda k: (k[0], aam_rank.get(k[1], 99))):
        ds_mf = sorted(groups[(thr, aam)].items())
        xs = [d for d, mf in ds_mf if keep(mf)]
        ys = [mf for d, mf in ds_mf if keep(mf)]
        if not xs:
            continue
        all_ys.extend(ys)
        ax.plot(xs, ys, thr_style[thr], markersize=5,
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
        # Dense x so the power law renders as a smooth curve on a linear y axis
        # (on a log y axis it is a straight line either way).
        xline = np.geomspace(fx.min(), fx.max(), 100)
        ax.plot(xline, C * xline ** k, color="black", alpha=0.5,
                linestyle=thr_fit_ls[thr], linewidth=1.5,
                label=f"fit acc>={thr}: {C:.3g}·d^{k:.2f}")

    if decorate:
        ax.set_xlabel("model size (d)")
        ax.set_ylabel(y_field)
        ax.set_title(f"{y_field} vs d  ({os.path.basename(path)})")

        # x ticks: the model sizes we sweep, as plain integers (no minor 10^x ticks).
        xticks = [16, 32, 64, 128, 256]
        ax.set_xticks(xticks)
        ax.set_xticks([], minor=True)
        ax.set_xticklabels([str(x) for x in xticks])

        # y ticks: powers of two spanning the plotted range (log axis only;
        # for a linear axis matplotlib's auto-ticks are more sensible).
        if y_log and all_ys:
            lo = int(np.floor(np.log2(min(all_ys))))
            hi = int(np.ceil(np.log2(max(all_ys))))
            yticks = [2 ** n for n in range(lo, hi + 1)]
            ax.set_yticks(yticks)
            ax.set_yticks([], minor=True)
            ax.set_yticklabels([str(y) for y in yticks])

        ax.grid(True, which="both", ls="--", linewidth=0.5)
        ax.legend(fontsize=9)

    return all_ys


def plot_capacity_vs_d(path=None, y_field="max_facts", y_log=True):
    """Log-x plot of y_field vs d, one line per (accuracy_threshold, any_all_most).

    y_field selects the y axis column: "max_facts" (default), "best_top_n", "best_S"
    or "best_top_fraction". y_log toggles the y-axis between log (default) and linear.

    Thin wrapper around _draw_capacity_vs_d: makes its own figure, shows it, and
    returns it. See that helper for the full description of what is drawn.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_capacity_vs_d(ax, path=path, y_field=y_field, y_log=y_log)
    fig.tight_layout()
    plt.show()
    return fig


def _find_e7_dir():
    """Locate the E7/ experiment folder (sibling of hand_coded_models/).

    Mirrors _find_results_dir: walk upward from this file and the cwd looking for a
    directory named E7.
    """
    candidates = []
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = None
    for start in [p for p in (here, os.getcwd()) if p]:
        cur = start
        while True:
            candidates.append(os.path.join(cur, "E7"))
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0] if candidates else "E7"


def _draw_e7(ax, experiment_dir=None, include=("simple", "full"), aam_color=None,
             decorate=True):
    """Draw the E7 max_facts-vs-d_residual plot onto a given Axes (no figure / show).

    Mirrors E7/E7_plot.py: one colored line per log file (color = any/most/all,
    style = model type), plus one black power-law fit per model-type group (pooling
    any/most/all), with the fitted formula in the legend. `include` keeps only log
    files whose name contains one of the listed model types (None = keep all).

    aam_color overrides the any/most/all colors. decorate=False skips the axis
    labels/ticks/title/legend (so a caller can overlay several datasets and set
    those once). Returns the list of plotted max_facts (for shared tick ranges).
    """
    if experiment_dir is None:
        experiment_dir = _find_e7_dir()

    log_files = [f[:-6] for f in os.listdir(experiment_dir)
                 if f.endswith(".jsonl") and f != "test_log.jsonl"
                 and os.path.getsize(os.path.join(experiment_dir, f)) > 0]
    if include is not None:
        log_files = [f for f in log_files if any(g in f for g in include)]

    # Colors match the hc2 capacity plot for any/most/all (unless overridden).
    if aam_color is None:
        aam_color = {"any": "tab:blue", "all": "tab:red", "most": "tab:green"}
    colors = aam_color
    styles = {"full": "-o", "simple": "--x", "nb": ":s"}
    fit_styles = {"full": "-", "simple": "--", "nb": ":"}
    group_order = {"simple": 0, "full": 1, "nb": 2}
    category_order = {"any": 0, "most": 1, "all": 2}

    def sort_key(name):
        group = next((v for k, v in group_order.items() if k in name), 99)
        category = next((v for k, v in category_order.items() if k in name), 99)
        return (group, category, name)
    log_files = sorted(log_files, key=sort_key)

    group_points = defaultdict(list)
    all_ys = []
    for log_file in log_files:
        # load_capacity_results is a generic JSONL reader (path + records list).
        runs = load_capacity_results(os.path.join(experiment_dir, log_file + ".jsonl"))
        ds = [run["settings"]["d_residual"] for run in runs]
        max_facts = [run["max_facts"] for run in runs]

        color = next((c for k, c in colors.items() if k in log_file), None)
        style = next((s for k, s in styles.items() if k in log_file), "-o")
        ax.loglog(ds, max_facts, style, color=color, label=log_file)

        all_ys.extend([mf for mf in max_facts if mf and mf > 0])
        group = next((g for g in group_order if g in log_file), None)
        for d, mf in zip(ds, max_facts):
            if mf and mf > 0:
                group_points[group].append((d, mf))

    # Power-law fits, one per model-type group (pooling any/most/all). y = C * d^k.
    for group in sorted(group_points, key=lambda g: group_order.get(g, 99)):
        pts = sorted(group_points[group])
        if len(pts) < 2:
            continue
        fx = np.array([d for d, _ in pts], dtype=float)
        fy = np.array([mf for _, mf in pts], dtype=float)
        k, b = np.polyfit(np.log(fx), np.log(fy), 1)
        C = np.exp(b)
        xline = np.array([fx.min(), fx.max()])
        ax.loglog(xline, C * xline ** k, color="black", alpha=0.5,
                  linestyle=fit_styles.get(group, "-"), linewidth=1.5,
                  label=f"fit {group}: {C:.3g}·d^{k:.2f}")

    if decorate:
        ax.set_xlabel("d_residual")
        ax.set_ylabel("max_facts")
        ax.set_title("E7: max_facts vs d_residual  (color = any/most/all, "
                     "style = model type; black = power-law fit)")

        xticks = [16, 32, 64, 128]
        ax.set_xticks(xticks)
        ax.set_xticks([], minor=True)
        ax.set_xticklabels([str(x) for x in xticks])
        if all_ys:
            lo = int(np.floor(np.log2(min(all_ys))))
            hi = int(np.ceil(np.log2(max(all_ys))))
            yticks = [2 ** n for n in range(lo, hi + 1)]
            ax.set_yticks(yticks)
            ax.set_yticks([], minor=True)
            ax.set_yticklabels([str(y) for y in yticks])

        ax.grid(True, which="both", ls="--", linewidth=0.5)
        ax.legend(fontsize=9)

    return all_ys


def plot_capacity_vs_d_and_e7(path=None, e7_dir=None, figsize=(11, 7)):
    """Overlay the hc2 capacity plot and the E7 plot on a single Axes.

    Both are max_facts vs model size (d / d_residual) on the same log-log axes. The
    hc2 lines keep the standard tab colors; the E7 lines are recolored (for this
    plot only) so the two datasets are distinguishable: red->orange, green->lime,
    blue->skyblue. Ticks span the combined range of both datasets.

    path defaults to capacity_search_results_topfrac.json — the FINE-grid
    top_fraction sweep (0.00-0.38 in 0.02 steps), the same log
    plot_capacity_vs_d_all_models uses for its hand-coded line. (The old default,
    capacity_search_results.json, holds the June-19 coarse 0.1-0.9 runs; pass it
    explicitly to reproduce the old plot.)
    """
    if path is None:
        path = os.path.join(RESULTS_DIR, "capacity_search_results_topfrac.json")
    fig, ax = plt.subplots(figsize=figsize)
    ys = _draw_capacity_vs_d(ax, path=path, decorate=False)

    # Recolor E7 (for this plot only): any(blue)->skyblue, all(red)->orange,
    # most(green)->lime.
    e7_color = {"any": "skyblue", "all": "orange", "most": "lime"}
    ys += _draw_e7(ax, experiment_dir=e7_dir, aam_color=e7_color, decorate=False)

    ax.set_xlabel("model size (d)")
    ax.set_ylabel("max facts")
    ax.set_title("Capacity vs model size: HandCodedModel2 (tab colors) "
                 "+ E7 (orange/lime/skyblue)")

    xticks = [16, 32, 64, 128, 256]
    ax.set_xticks(xticks)
    ax.set_xticks([], minor=True)
    ax.set_xticklabels([str(x) for x in xticks])
    if ys:
        lo = int(np.floor(np.log2(min(ys))))
        hi = int(np.ceil(np.log2(max(ys))))
        yticks = [2 ** n for n in range(lo, hi + 1)]
        ax.set_yticks(yticks)
        ax.set_yticks([], minor=True)
        ax.set_yticklabels([str(y) for y in yticks])

    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    plt.show()
    return fig


def plot_capacity_vs_d_all_models(thresholds=(0.9, 1.0),
                                  any_all_most=("any", "most", "all"),
                                  results_dir=None, figsize=(11, 7),
                                  y_field="max_facts", models=None,
                                  ylabel=None, title=None, references=None,
                                  y_log=True, yticks=None, fit_type="power",
                                  point_spread=0.0, connect_points=True,
                                  thr_markers=None):
    """Overlay the capacity-vs-d curves of all four model variants on one log-log plot.

    Reads the four capacity-search JSONL logs (skipping any that don't exist yet,
    e.g. sweeps still running):
        capacity_search_results_topfrac.json         hand-coded up + hand-coded down
        capacity_search_results_topfrac_hybrid.json  hand-coded up + TRAINED down
        capacity_search_results_randomup.json        RANDOM frozen up + trained down
        capacity_search_results_fulltrain.json       whole network trained

    Encoding (color = model variant, since that is the comparison here):
        color      : model variant (trained blue, hybrid orange,
                     hand-coded green, rand-emb red)
        line style : accuracy_threshold (solid = 1.0, dashed = 0.9)
        marker     : any_all_most (any = o, most = ^, all = x)

    Power-law fits (max_facts = C * d^k, straight lines in log-log space) are drawn
    per (model, accuracy_threshold), pooling the any/most/all points, in the model's
    color (translucent, no markers) with the fitted formula in the legend — the same
    pooling plot_capacity_vs_d_and_e7's components use.

    thresholds / any_all_most filter what is drawn (the full 4x2x3 legend is busy;
    e.g. any_all_most=("most",) gives one line per model and threshold). Rows with
    y_field <= 0 are dropped (they can't appear on a log axis); if a log holds
    several rows for the same (d, threshold, rule), the latest wins.

    y_field: which log column goes on the y axis ("max_facts" default; e.g.
    "best_S" — only the hand-coded and hybrid logs have S). models: which model
    variants to include, by name (default: all four). ylabel / title: override
    the default axis label and title. references: optional list of
    (func, label, color) hypothesis lines y = func(d), drawn across the data's
    x range. connect_points=False draws the data as markers only (no lines
    between the points; the fit lines stay). thr_markers: optional
    {threshold: marker} dict that makes the marker encode the threshold
    instead of any/most/all — useful when only one rule is drawn.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    # Order sets the legend: trained, hybrid, hand-coded, rand-emb
    # (roughly strongest to weakest, so the legend reads top-down like the plot).
    model_files = [
        ("trained", "capacity_search_results_fulltrain.json", "tab:blue"),
        ("hybrid", "capacity_search_results_topfrac_hybrid.json", "tab:orange"),
        ("hand-coded", "capacity_search_results_topfrac.json", "tab:green"),
        ("rand-emb", "capacity_search_results_randomup.json", "tab:red"),
    ]
    if models is not None:
        # Keep the caller's order: `models` sets legend/draw order, not just the filter.
        by_name = {mf[0]: mf for mf in model_files}
        model_files = [by_name[m] for m in models if m in by_name]
    thr_style = {1.0: "-", 0.9: "--"}
    aam_marker = {"any": "o", "most": "^", "all": "x"}
    aam_rank = {"any": 0, "most": 1, "all": 2}

    def thr_label(thr):
        """Legend text for a threshold: 'acc=1' for 1.0, 'acc≥thr' otherwise."""
        return "acc=1" if thr == 1.0 else fr"acc$\geq${thr}"

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale("log")
    ax.set_yscale("log" if y_log else "linear")

    # On a log y axis, zeros/negatives can't be shown, so those rows are
    # dropped; on a linear axis (y_log=False) they are kept — only missing
    # values are dropped. Fits always use the positive points only.
    keep = (lambda v: v is not None and v > 0) if y_log \
        else (lambda v: v is not None)

    all_xs, all_ys = [], []
    draw_queue = []  # ("data"/"fit", kwargs) in legend order; drawn after
    for model_name, fname, color in model_files:
        path = os.path.join(results_dir, fname)
        runs = load_capacity_results(path)
        if not runs:
            print(f"(skipping {model_name}: no results in {fname})")
            continue

        # Group rows by (threshold, rule) -> {d: y_field}; the latest row wins.
        groups = defaultdict(dict)
        for r in runs:
            if r.get("accuracy_threshold") in thresholds \
                    and r.get("any_all_most") in any_all_most:
                groups[(r["accuracy_threshold"], r["any_all_most"])][r["d"]] = r.get(y_field)

        thr_points = defaultdict(list)  # threshold -> pooled (d, y_field) for the fit
        for (thr, aam) in sorted(groups, key=lambda k: (k[0], aam_rank.get(k[1], 99))):
            ds_mf = sorted(groups[(thr, aam)].items())
            xs = [d for d, mf in ds_mf if keep(mf)]
            ys = [mf for d, mf in ds_mf if keep(mf)]
            if not xs:
                continue
            all_xs.extend(xs)
            all_ys.extend(ys)
            # A power-law fit needs log(y), so it can only use positive points;
            # the linlog fit (y linear in log2(d)) handles zeros fine.
            thr_points[thr].extend(
                (x, y) for x, y in zip(xs, ys)
                if fit_type == "linlog" or y > 0)
            marker = thr_markers.get(thr) if thr_markers \
                else aam_marker.get(aam)
            draw_queue.append(("data", dict(
                xs=xs, ys=ys,
                fmt=thr_style.get(thr, ":") if connect_points else "none",
                marker=marker, color=color,
                label=f"{model_name}: {aam}, {thr_label(thr)}")))

        # One fit per threshold for this model, pooling any/most/all.
        for thr in sorted(thr_points):
            pts = sorted(thr_points[thr])
            if len({d for d, _ in pts}) < 2:
                continue
            fx = np.array([d for d, _ in pts], dtype=float)
            fy = np.array([mf for _, mf in pts], dtype=float)
            # Dense x so the fit renders as a smooth curve whatever the axes.
            xline = np.geomspace(fx.min(), fx.max(), 100)
            if fit_type == "linlog":
                # y = a*log2(d) + b — linear in log model size; unlike a power
                # law it can pass through (and be fit to) zero values.
                a, b = np.polyfit(np.log2(fx), fy, 1)
                yline = a * np.log2(xline) + b
                fit_label = (f"fit {model_name} {thr_label(thr)}: "
                             f"{a:.3g}·log2(d) {b:+.3g}")
            else:  # power law over log: y = C * d^k / log(d)
                # ln(y) + ln(ln(d)) = ln(C) + k·ln(d), a plain linear fit of
                # (ln y + ln ln d) against ln d.
                k, b = np.polyfit(np.log(fx), np.log(fy) + np.log(np.log(fx)), 1)
                C = np.exp(b)
                yline = C * xline ** k / np.log(xline)
                fit_label = f"fit {model_name} {thr_label(thr)}: {C:.3g}·d^{k:.2f}/log(d)"
            draw_queue.append(("fit", dict(
                xline=xline, yline=yline, color=color,
                ls=thr_style.get(thr, ":"), label=fit_label)))

    # Fan out data points that several lines share: points with the exact same
    # (d, y) get small symmetric x offsets (a step in log10 units, like the
    # scatter plots' spread), so coinciding lines/markers stay visible.
    # point_spread=0 (the default) keeps every point exactly at its true d.
    if point_spread:
        occurrences = defaultdict(list)  # (d, y) -> [(queue idx, point idx)]
        for qi, (kind, kw) in enumerate(draw_queue):
            if kind == "data":
                for pi, xy in enumerate(zip(kw["xs"], kw["ys"])):
                    occurrences[xy].append((qi, pi))
        for (x, _y), locs in occurrences.items():
            if len(locs) < 2:
                continue
            offsets = np.arange(len(locs)) - (len(locs) - 1) / 2.0
            for off, (qi, pi) in zip(offsets, locs):
                kw = draw_queue[qi][1]
                kw.setdefault("x_plot", list(kw["xs"]))
                kw["x_plot"][pi] = x * 10.0 ** (off * point_spread)

    for kind, kw in draw_queue:
        if kind == "data":
            ax.plot(kw.get("x_plot", kw["xs"]), kw["ys"], linestyle=kw["fmt"],
                    marker=kw["marker"], markersize=5, color=kw["color"],
                    label=kw["label"])
        else:
            ax.plot(kw["xline"], kw["yline"], color=kw["color"], alpha=0.4,
                    linestyle=kw["ls"], linewidth=3, label=kw["label"])

    # Optional hypothesis lines y = func(d), drawn across the data's x range.
    if references and all_xs:
        xline = np.geomspace(min(all_xs), max(all_xs), 100)
        for func, ref_label, ref_color in references:
            ax.plot(xline, func(xline), color=ref_color, lw=1.8, label=ref_label)

    if ylabel is None:
        ylabel = "max facts" if y_field == "max_facts" else y_field
    if title is None:
        title = ("Capacity vs model size\n"
                 + " / ".join(name for name, _, _ in model_files))
    ax.set_xlabel("model size (d)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    xticks = [x for x in [16, 32, 64, 128, 256] if not all_xs or x <= max(all_xs)]
    ax.set_xticks(xticks)
    ax.set_xticks([], minor=True)
    ax.set_xticklabels([str(x) for x in xticks])
    if yticks is not None:
        # Explicit y ticks (and so horizontal gridlines) from the caller,
        # trimmed to the range the data actually covers.
        if all_ys:
            yticks = [t for t in yticks if min(all_ys) <= t <= max(all_ys)]
        ax.set_yticks(yticks)
        ax.set_yticks([], minor=True)
    elif y_log and all_ys:
        # Powers-of-two y ticks (log axis only; linear keeps auto ticks).
        lo = int(np.floor(np.log2(min(all_ys))))
        hi = int(np.ceil(np.log2(max(all_ys))))
        yticks = [2 ** n for n in range(lo, hi + 1)]
        ax.set_yticks(yticks)
        ax.set_yticks([], minor=True)
        ax.set_yticklabels([str(y) for y in yticks])

    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    plt.show()
    return fig


def plot_capacity_vs_d_any_only(thresholds=(0.9, 1.0), results_dir=None,
                                figsize=(8, 6), models=None, error_bars=True):
    """Capacity vs d for all model variants, "any" rule only (log-log).

    Simplified variant of plot_capacity_vs_d_all_models, styled like the
    "any"-only plot in E7_plot.py: data is drawn as markers only (no
    connecting lines), the marker encodes the accuracy threshold
    (acc=1 -> o, acc>=0.9 -> x) since only one rule is shown, and each
    power-law fit ("best fit: C·d^k", fitted to just that series' points)
    follows its data series in the legend — data, fit, data, fit, ...

    error_bars=True gives each point a one-sided (upward) error bar of that
    run's logged binary-search `precision`: the reported max_facts is a
    confirmed success, and the true capacity can be up to ~precision higher.
    The bars are drawn separately from the markers and stay out of the
    legend, so they don't change the legend or the layout.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    # Same four logs, colors and legend order as plot_capacity_vs_d_all_models.
    model_files = [
        ("trained", "capacity_search_results_fulltrain.json", "tab:blue"),
        ("hybrid", "capacity_search_results_topfrac_hybrid.json", "tab:orange"),
        ("hand-coded", "capacity_search_results_topfrac.json", "tab:green"),
        ("rand-emb", "capacity_search_results_randomup.json", "tab:red"),
    ]
    if models is not None:
        by_name = {mf[0]: mf for mf in model_files}
        model_files = [by_name[m] for m in models if m in by_name]
    thr_style = {1.0: "-", 0.9: "--"}
    thr_marker = {1.0: "o", 0.9: "x"}

    def thr_label(thr):
        """Legend text for a threshold: 'acc=1' for 1.0, 'acc≥thr' otherwise."""
        return "acc=1" if thr == 1.0 else fr"acc$\geq${thr}"

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale("log")
    ax.set_yscale("log")

    all_xs, all_ys = [], []
    for model_name, fname, color in model_files:
        runs = load_capacity_results(os.path.join(results_dir, fname))
        if not runs:
            print(f"(skipping {model_name}: no results in {fname})")
            continue
        for thr in sorted(thresholds):
            # Latest row per d wins; zeros can't show on the log axis.
            by_d = {}
            for r in runs:
                if r.get("accuracy_threshold") == thr \
                        and r.get("any_all_most") == "any":
                    by_d[r["d"]] = (r.get("max_facts"), r.get("precision", 0))
            pts = sorted((d, mf, p) for d, (mf, p) in by_d.items()
                         if mf and mf > 0)
            if not pts:
                continue
            xs = [d for d, _, _ in pts]
            ys = [mf for _, mf, _ in pts]
            all_xs.extend(xs)
            all_ys.extend(ys)
            ax.plot(xs, ys, linestyle="none", marker=thr_marker.get(thr, "o"),
                    markersize=6, color=color,
                    label=f"{model_name}: {thr_label(thr)}")
            if error_bars:
                # precision 1 means the search resolved max_facts exactly.
                errs = [0 if p <= 1 else p for _, _, p in pts]
                ax.errorbar(xs, ys, yerr=[np.zeros(len(ys)), errs],
                            fmt="none", ecolor=color, capsize=3,
                            label="_nolegend_")

            # Fit a·d^b/log(d) to just this series, after it in the legend.
            # Taking logs: ln(y) + ln(ln(d)) = ln(a) + b·ln(d), a plain
            # linear fit of (ln y + ln ln d) against ln d.
            if len(set(xs)) < 2:
                continue
            fx = np.array(xs, dtype=float)
            fy = np.array(ys, dtype=float)
            b, loga = np.polyfit(np.log(fx), np.log(fy) + np.log(np.log(fx)), 1)
            a = np.exp(loga)
            xline = np.linspace(fx.min(), fx.max(), 100)
            ax.plot(xline, a * xline ** b / np.log(xline), color=color,
                    alpha=0.4, linestyle=thr_style.get(thr, ":"), linewidth=3,
                    label=f"best fit: {a:.3g}·d^{b:.2f}/log(d)")

    ax.set_xlabel("model size (d)")
    ax.set_ylabel("max facts")
    ax.set_title("Capacity vs model size\n"
                 + " / ".join(name for name, _, _ in model_files))

    xticks = [x for x in [16, 32, 64, 128, 256] if not all_xs or x <= max(all_xs)]
    ax.set_xticks(xticks)
    ax.set_xticks([], minor=True)
    ax.set_xticklabels([str(x) for x in xticks])
    if all_ys:
        lo = int(np.floor(np.log2(min(all_ys))))
        hi = int(np.ceil(np.log2(max(all_ys))))
        yticks = [2 ** n for n in range(lo, hi + 1)]
        ax.set_yticks(yticks)
        ax.set_yticks([], minor=True)
        ax.set_yticklabels([str(y) for y in yticks])

    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    plt.show()
    return fig


def plot_best_S_vs_d(thresholds=(0.9, 1.0), any_all_most=("any", "most", "all"),
                     results_dir=None, figsize=(11, 7), point_spread=0.008):
    """best_S vs model size d, hand-coded and hybrid only (log-log).

    Same layout/encoding as plot_capacity_vs_d_all_models (color = model, line
    style = threshold, marker = any/most/all, translucent power-law fits), just
    with best_S on the y axis. Only these two variants sweep S — the trained and
    rand-emb models have no S at all — so only their logs are drawn.
    """
    return plot_capacity_vs_d_all_models(
        thresholds=thresholds, any_all_most=any_all_most,
        results_dir=results_dir, figsize=figsize,
        y_field="best_S", models=("hand-coded", "hybrid"),
        ylabel="best S",
        # Several lines often pick the same (d, S); fan those points out
        # slightly in x so the coinciding lines stay visible (0 turns it off).
        point_spread=point_spread,
        title="Best performing S vs model size\nhand-coded / hybrid",
        references=[
            (np.sqrt, r"$\sqrt{d}$", "black"),
        ])


def plot_best_top_fraction_vs_d(thresholds=(0.9, 1.0),
                                any_all_most=("any", "most", "all"),
                                results_dir=None, figsize=(11, 7),
                                point_spread=0.008):
    """best_top_fraction vs model size d, hand-coded and hybrid only.

    Same layout/encoding as plot_best_S_vs_d (color = model, line style =
    threshold, marker = any/most/all, translucent power-law fits), but with the
    winning top_fraction on the y axis — on a LINEAR scale, since
    best_top_fraction can be 0, which a log axis cannot show (x stays log).
    Only the hand-coded and hybrid logs have a top_fraction.
    """
    return plot_capacity_vs_d_all_models(
        thresholds=thresholds, any_all_most=any_all_most,
        results_dir=results_dir, figsize=figsize,
        y_field="best_top_fraction", models=("hand-coded", "hybrid"),
        ylabel="best top_fraction", y_log=False,
        # Horizontal gridlines at exactly the swept top_fraction values.
        yticks=[round(0.02 * i, 2) for i in range(20)],
        # Fit y = a*log2(d) + b: unlike a power law it can use the tf=0 points.
        fit_type="linlog",
        # Several lines often pick the same (d, top_fraction); fan those points
        # out slightly in x so the coinciding lines stay visible (0 turns it off).
        point_spread=point_spread,
        title="Best performing top_fraction vs model size\nhand-coded / hybrid")


def plot_decoupled_capacity_grids(accuracy_threshold, any_all_most,
                                  input_vocab_sizes=(16, 32), path=None):
    """2x2 heatmap grid from capacity_search_results_topfrac_decoupled.json.

    Layout (for the default input_vocab_sizes=(16, 32)):
        columns : input_vocab_size   (left = 16, right = 32)
        top row    -> max_facts       (each square annotated with the max_facts number)
        bottom row -> best_S          (each square annotated with the best_S number)

    In every subplot the x-axis is d_ff and the y-axis is output_vocab_size. A row
    that is generated by the decoupled capacity search has one record per
    (input_vocab_size, output_vocab_size, d_ff, accuracy_threshold, any_all_most),
    so accuracy_threshold and any_all_most must be given to pick a single value per
    square. Color is scaled per row (shared across the columns) so the two
    input_vocab_size panels in a row are directly comparable.
    """
    if path is None:
        path = os.path.join(RESULTS_DIR, "capacity_search_results_topfrac_decoupled.json")
    runs = load_capacity_results(path)
    if not runs:
        raise FileNotFoundError(f"No capacity results found in {path}")

    # Keep only rows matching the requested success criterion and the wanted columns.
    sel = [r for r in runs
           if r.get("accuracy_threshold") == accuracy_threshold
           and r.get("any_all_most") == any_all_most
           and r.get("input_vocab_size") in input_vocab_sizes]
    if not sel:
        raise ValueError(
            f"No rows for accuracy_threshold={accuracy_threshold}, "
            f"any_all_most={any_all_most}, input_vocab_size in {input_vocab_sizes} "
            f"(in {os.path.basename(path)})"
        )

    # Shared axes across all subplots: d_ff on x, output_vocab_size on y.
    d_ff_values = sorted({r["d_ff"] for r in sel})
    ov_values = sorted({r["output_vocab_size"] for r in sel})
    d_idx = {d: j for j, d in enumerate(d_ff_values)}
    o_idx = {o: i for i, o in enumerate(ov_values)}

    def grid_for(iv, field):
        """(output_vocab_size x d_ff) array of `field` for one input_vocab_size."""
        arr = np.full((len(ov_values), len(d_ff_values)), np.nan)
        for r in sel:
            if r["input_vocab_size"] == iv:
                arr[o_idx[r["output_vocab_size"]], d_idx[r["d_ff"]]] = r[field]
        return arr

    fields = ["max_facts", "best_S"]  # top row, bottom row
    n_col = len(input_vocab_sizes)
    fig, axes = plt.subplots(2, n_col, figsize=(3.4 * n_col + 1, 7), squeeze=False)

    for row, field in enumerate(fields):
        grids = [grid_for(iv, field) for iv in input_vocab_sizes]
        # Shared color scale per row so the columns are comparable.
        finite = np.concatenate([g[np.isfinite(g)].ravel() for g in grids]
                                or [np.array([])])
        vmin = float(finite.min()) if finite.size else 0.0
        vmax = float(finite.max()) if finite.size else 1.0
        for col, iv in enumerate(input_vocab_sizes):
            ax = axes[row][col]
            grid = grids[col]
            im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                           vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(d_ff_values)))
            ax.set_xticklabels(d_ff_values)
            ax.set_yticks(range(len(ov_values)))
            ax.set_yticklabels(ov_values)
            ax.set_xlabel("d_ff")
            ax.set_ylabel("output_vocab_size")
            ax.set_title(f"{field}  |  input_vocab_size={iv}")

            # Annotate each square with its number (blank where there is no data).
            for i in range(len(ov_values)):
                for j in range(len(d_ff_values)):
                    v = grid[i, j]
                    if np.isnan(v):
                        continue
                    frac = 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)
                    ax.text(j, i, f"{int(round(v))}", ha="center", va="center",
                            color="white" if frac < 0.5 else "black", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Decoupled capacity grids  |  accuracy_threshold={accuracy_threshold}, "
        f"any_all_most={any_all_most}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    return fig


def _scatter_best_S_vs(x_field, path=None, spread=0.012, reference=None,
                       legend_between=None, ax=None, figsize=(8, 6),
                       xlabel=None, title=None,
                       y_field="best_S", y_log=True, yticks=None, ylabel=None,
                       fit_xy_label=False, fit_x_name=None,
                       legend_framealpha=None):
    """Log-log scatter of best_S vs `x_field` for every row in the decoupled log.

    One point per record in capacity_search_results_topfrac_decoupled.json (all
    input_vocab_size / output_vocab_size / accuracy_threshold / any_all_most rows
    are shown together), with a least-squares power-law best fit.

    x_field and best_S only take a handful of discrete values, so many rows land
    on the exact same spot. To show how many, the rows sharing an (x, best_S) are
    laid out in a neat horizontal row, centered on their true x and spaced by
    `spread` (a step in log10 units, so partly-overlapping on the log x-axis). y
    stays at the true best_S. Set spread=0 to stack them exactly.

    reference: optional (func, label) drawing a hypothesis line y=func(x) in black.
    The fit and the reference always use the raw (un-fanned) values.

    legend_between: optional (y_low, y_high) placing the legend, left-aligned,
    centered vertically in that y-band (data coords). Handy when a corner of the
    plot is empty. None uses matplotlib's default best placement.

    ax: draw onto this Axes instead of making a new figure (used by the 2x2 grid).
    When ax is None a standalone figure is created and shown.

    xlabel / title: override the default x-axis label (x_field) and the default
    auto-generated title. None keeps the defaults.

    y_field / y_log / yticks / ylabel: which log column goes on the y axis
    (default best_S), whether the y axis is log (default) or linear, explicit y
    ticks (trimmed to the data range; None keeps the y_field-specific default),
    and the y-axis label. On a log y axis non-positive rows are dropped and the
    fit is a power law; on a linear axis zeros are kept and the fit is
    y = a*log2(x) + b (which can pass through zero).
    """
    if path is None:
        path = os.path.join(RESULTS_DIR, "capacity_search_results_topfrac_decoupled.json")
    runs = load_capacity_results(path)
    if not runs:
        raise FileNotFoundError(f"No capacity results found in {path}")

    x = np.array([r[x_field] for r in runs], dtype=float)
    yv = np.array([np.nan if r.get(y_field) is None else r[y_field]
                   for r in runs], dtype=float)
    # A log y axis can't show non-positive values, so they are dropped there;
    # on a linear axis (y_log=False) zeros are kept — only missing are dropped.
    ok = np.isfinite(x) & np.isfinite(yv) & (x > 0)
    if y_log:
        ok &= yv > 0
    x, yv = x[ok], yv[ok]

    # LaTeX-safe name for the formula labels: fit_x_name if given, else the
    # xlabel override, else the raw field name.
    math_x = (fit_x_name or (x_field if xlabel is None else xlabel)).replace("_", r"\_")

    own_fig = ax is None
    fig, ax = (plt.subplots(figsize=figsize) if own_fig else (ax.figure, ax))
    ax.set_xscale("log")
    ax.set_yscale("log" if y_log else "linear")
    # Fan out the rows that share an (x, y): line them up horizontally,
    # centered on the true x, spaced evenly in log space so they partly overlap.
    x_plot = x.copy()
    groups = defaultdict(list)
    for idx, key in enumerate(zip(x, yv)):
        groups[key].append(idx)
    for (xv, _sy), idxs in groups.items():
        n = len(idxs)
        offsets = np.arange(n) - (n - 1) / 2.0  # symmetric: e.g. -1,0,1
        for off, idx in zip(offsets, idxs):
            x_plot[idx] = xv * 10.0 ** (off * spread)
    ax.scatter(x_plot, yv, s=25, alpha=0.5, edgecolor="none",
               color="tab:blue")

    xline = np.geomspace(x.min(), x.max(), 100)

    # Optional hypothesis line.
    if reference is not None:
        ref_func, ref_label = reference
        ax.plot(xline, ref_func(xline), color="black", lw=1.8, label=ref_label)

    math_y = y_field.replace("_", r"\_")
    if y_log:
        # Best-fit power law y = C * x^k (a straight line in log-log space).
        k, b = np.polyfit(np.log(x), np.log(yv), 1)
        C = np.exp(b)
        yline = C * xline ** k
        if fit_xy_label:
            # Plain "y = ... x ..." with 2 significant digits.
            fit_label = fr"Best fit: $y = {C:.2g}\cdot x^{{{k:.2g}}}$"
        else:
            fit_label = fr"Best fit: ${math_y} = {C:.3g}\cdot {math_x}^{{{k:.2f}}}$"
    else:
        # y = a*log2(x) + b — linear in log x; unlike a power law it can pass
        # through (and be fit to) zero values.
        a, b = np.polyfit(np.log2(x), yv, 1)
        yline = a * np.log2(xline) + b
        if fit_xy_label:
            # Plain "y = ... log2(x) ..." with 2 significant digits.
            fit_label = fr"Best fit: $y = {a:.2g}\cdot\log_2(x) {b:+.2g}$"
        else:
            fit_label = fr"Best fit: ${math_y} = {a:.3g}\cdot\log_2({math_x}) {b:+.3g}$"
    ax.plot(xline, yline, color="tab:red", lw=1.8, linestyle="--",
            label=fit_label)

    ax.set_xlabel(x_field if xlabel is None else xlabel)
    if ylabel is None:
        ylabel = "best S" if y_field == "best_S" else y_field
    ax.set_ylabel(ylabel)
    if title is None:
        title = f"{y_field} vs {x_field}  ({os.path.basename(path)}, {len(x)} points)"
    ax.set_title(title)

    # Integer ticks that match the swept values.
    xticks = sorted({int(v) for v in np.unique(x)})
    ax.set_xticks(xticks)
    ax.set_xticks([], minor=True)
    ax.set_xticklabels([str(v) for v in xticks])
    if yticks is not None:
        # Explicit y ticks from the caller, trimmed to the data's range.
        ax.set_yticks([t for t in yticks if yv.min() <= t <= yv.max()])
        ax.set_yticks([], minor=True)
    elif y_log:
        # Tick every integer in the data's range (not just the values present,
        # so gridline rows stay evenly spaced even when some value never
        # occurs) ...
        yticks = list(range(int(yv.min()), int(yv.max()) + 1))
        ax.set_yticks(yticks)
        ax.set_yticks([], minor=True)
        # ... but only print the labels below — consecutive integers overlap
        # on the log axis otherwise.
        labels_to_keep = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22}
        ax.set_yticklabels([str(v) if v in labels_to_keep else "" for v in yticks])

    ax.grid(True, which="both", ls="--", linewidth=0.5)
    # legend_framealpha: legend background opacity (None keeps the defaults).
    if legend_between is not None:
        y_low, y_high = legend_between
        # Center of the band: geometric on a log y axis, arithmetic on linear.
        y_center = (y_low * y_high) ** 0.5 if y_log else (y_low + y_high) / 2.0
        ax.legend(loc="center left",
                  framealpha=0.9 if legend_framealpha is None else legend_framealpha,
                  bbox_to_anchor=(x.min(), y_center), bbox_transform=ax.transData)
    else:
        ax.legend(framealpha=legend_framealpha)
    if own_fig:
        fig.tight_layout()
        plt.show()
    return fig


def plot_best_S_vs_d_ff(path=None, spread=0.012, figsize=(8, 6)):
    """best_S vs d_ff (log-log), with the best_S = sqrt(d_ff) hypothesis line and
    a power-law best fit. See _scatter_best_S_vs for the layout details."""
    variant = "hand-coded model" if path is None else "hybrid model"
    return _scatter_best_S_vs(
        "d_ff", path=path, spread=spread, figsize=figsize,
        xlabel="hidden layer size (d_MLP)", fit_x_name="d_MLP",
        title="Best performing S vs hidden layer size\n" + variant,
        reference=(np.sqrt, r"Hypotesis: $best\_S = \sqrt{d\_MLP}$"))


def plot_best_S_vs_input_vocab(path=None, spread=0.0087, ax=None):
    """best_S vs input_vocab_size (log-log), with a power-law best fit only (no
    hypothesis line). See _scatter_best_S_vs for the layout details.

    Default spread is tuned so the on-screen dot spacing matches
    plot_best_S_vs_output_vocab (its narrower x-range needs a smaller spread)."""
    return _scatter_best_S_vs("input_vocab_size", path=path, spread=spread,
                              legend_between=(1, 2), ax=ax)


def plot_best_S_vs_output_vocab(path=None, spread=0.012, ax=None):
    """best_S vs output_vocab_size (log-log), with a power-law best fit only (no
    hypothesis line). See _scatter_best_S_vs for the layout details."""
    return _scatter_best_S_vs("output_vocab_size", path=path, spread=spread,
                              legend_between=(1, 2), ax=ax)


def _strip_best_S_by(x_field, path=None, order=None, spread=0.06, ax=None,
                     xlabel=None, ylog=False,
                     y_field="best_S", yticks=None, ylabel=None):
    """Strip plot of best_S grouped by a categorical `x_field`.

    For a discrete x (e.g. accuracy_threshold with 2 values, or any_all_most with
    3 categories) a fit is not meaningful, so this just shows the raw datapoints:
    every row is plotted at its category's x-position, with no fitted/summary line.

    As in the log-log scatters, rows sharing a (category, best_S) are fanned out
    into a neat horizontal row (here linearly, since x is categorical) spaced by
    `spread` so overlapping rows are visible. `order` fixes the category order on
    the x-axis; default is sorted unique values.

    xlabel overrides the x-axis label (pass "" to hide it; None uses x_field).
    ylog: log-scale the y axis, with the same integer ticks / thinned labels as
    the log-log scatters (so the panel matches them in a grid).
    ax: draw onto this Axes instead of making a new figure (used by the 2x2 grid).
    y_field / yticks / ylabel: which log column goes on the y axis (default
    best_S), explicit y ticks (trimmed to the data range; for linear axes, e.g.
    top_fraction), and the y-axis label.
    """
    if path is None:
        path = os.path.join(RESULTS_DIR, "capacity_search_results_topfrac_decoupled.json")
    runs = load_capacity_results(path)
    if not runs:
        raise FileNotFoundError(f"No capacity results found in {path}")

    # Keep rows with a usable y value; category values may be numbers or strings.
    rows = [(r[x_field], float(r[y_field])) for r in runs
            if r.get(y_field) is not None and np.isfinite(float(r[y_field]))]
    if not rows:
        raise ValueError(f"No usable {y_field} rows for x_field={x_field}")

    cats = order if order is not None else sorted({c for c, _ in rows})
    pos = {c: i for i, c in enumerate(cats)}

    own_fig = ax is None
    fig, ax = (plt.subplots(figsize=(8, 6)) if own_fig else (ax.figure, ax))

    # Fan out rows sharing a (category, best_S): line them up horizontally around
    # the category's integer x-position, spaced linearly so they partly overlap.
    by_cell = defaultdict(list)
    for c, s in rows:
        if c in pos:
            by_cell[(c, s)].append(s)
    for (c, s), vals in by_cell.items():
        n = len(vals)
        offsets = (np.arange(n) - (n - 1) / 2.0) * spread
        ax.scatter(pos[c] + offsets, [s] * n, s=25, alpha=0.5,
                   edgecolor="none", color="tab:blue")

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([str(c) for c in cats])
    ax.set_xlim(-0.5, len(cats) - 0.5)
    ax.set_xlabel(x_field if xlabel is None else xlabel)
    ax.set_ylabel(("best S" if y_field == "best_S" else y_field)
                  if ylabel is None else ylabel)
    ax.set_title(f"{y_field} vs {x_field}  ({os.path.basename(path)}, {len(rows)} points)")
    if yticks is not None:
        # Explicit y ticks from the caller, trimmed to the data's range.
        yv = [s for _, s in rows]
        ax.set_yticks([t for t in yticks if min(yv) <= t <= max(yv)])
        ax.set_yticks([], minor=True)
    elif ylog:
        ax.set_yscale("log")
        svals = [int(s) for _, s in rows]
        # Tick every integer in the range (not just values present), so the
        # gridline rows stay evenly spaced.
        yticks = list(range(min(svals), max(svals) + 1))
        ax.set_yticks(yticks)
        ax.set_yticks([], minor=True)
        labels_to_keep = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22}
        ax.set_yticklabels([str(v) if v in labels_to_keep else "" for v in yticks])
    ax.grid(True, axis="y", ls="--", linewidth=0.5)
    if own_fig:
        fig.tight_layout()
        plt.show()
    return fig


def plot_best_S_vs_accuracy_threshold(path=None, spread=0.0197, ax=None, ylog=False):
    """Strip plot of best_S grouped by accuracy_threshold (raw datapoints only).
    See _strip_best_S_by for details. Default spread is tuned so the on-screen dot
    spacing matches plot_best_S_vs_output_vocab."""
    return _strip_best_S_by("accuracy_threshold", path=path, spread=spread, ax=ax,
                            ylog=ylog)


def plot_best_S_vs_any_all_most(path=None, spread=0.0296, ax=None, ylog=False):
    """Strip plot of best_S grouped by any/most/all (raw datapoints only).
    See _strip_best_S_by for details. Default spread is tuned so the on-screen dot
    spacing matches plot_best_S_vs_output_vocab. The x-axis label is hidden (the
    any/most/all tick labels already name the categories)."""
    return _strip_best_S_by("any_all_most", path=path, spread=spread,
                            order=["any", "most", "all"], ax=ax, xlabel="",
                            ylog=ylog)


def plot_best_S_grid(path=None, figsize=(15, 11)):
    """2x2 grid of the four best_S plots: input_vocab_size & output_vocab_size (top),
    accuracy_threshold & any_all_most (bottom). Reuses each plot's own drawing via
    its ax argument, so the panels match the standalone versions — except that in
    the grid, all four y axes are log (ylog=True for the bottom strip plots) and
    the x-axis labels are spelled out in plain English."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    plot_best_S_vs_input_vocab(path=path, ax=axes[0][0])
    plot_best_S_vs_output_vocab(path=path, ax=axes[0][1])
    plot_best_S_vs_accuracy_threshold(path=path, ax=axes[1][0], ylog=True)
    plot_best_S_vs_any_all_most(path=path, ax=axes[1][1], ylog=True)

    # Plain-English x-axis labels ("" is the any/most/all panel, which hides its
    # label when standalone; in the grid it gets a descriptive one).
    plain_xlabels = {
        "input_vocab_size": "input vocab size",
        "output_vocab_size": "output vocab size",
        "accuracy_threshold": "required accuracy",
        "": "how many attempts must succeed",
    }
    # y numbers printed in the grid (every value keeps its tick/gridline; this
    # only thins the labels, and only here — the standalone plots are untouched).
    grid_y_labels = {1, 2, 3, 4, 5, 6, 8, 10, 14, 18, 22}
    for a in axes.ravel():  # drop the per-panel titles + enlarge text in the grid
        a.set_title("")
        a.set_xlabel(plain_xlabels.get(a.get_xlabel(), a.get_xlabel()))
        a.set_yticklabels([str(int(v)) if int(v) in grid_y_labels else ""
                           for v in a.get_yticks()])
        a.xaxis.label.set_size(15)
        a.yaxis.label.set_size(15)
        a.tick_params(labelsize=13)
        leg = a.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_fontsize(12)
    variant = "hand-coded model" if path is None else "hybrid model"
    fig.suptitle("Best performing S vs everything else\n" + variant,
                 fontsize=18, y=0.995)
    fig.tight_layout()
    # tight_layout ignores the suptitle; set the axes top explicitly so the
    # panels start right below the two title lines.
    fig.subplots_adjust(top=0.925)
    plt.show()
    return fig


def plot_best_top_fraction_vs_d_ff(path=None, spread=0.012, figsize=(8, 6)):
    """best_top_fraction vs d_ff, like plot_best_S_vs_d_ff but with a LINEAR y
    axis (top_fraction can be 0, which a log axis can't show), gridlines at the
    swept top_fraction values, and a y = a*log2(d_MLP) + b best fit."""
    variant = "hand-coded model" if path is None else "hybrid model"
    return _scatter_best_S_vs(
        "d_ff", path=path, spread=spread, figsize=figsize,
        y_field="best_top_fraction", y_log=False,
        yticks=[round(0.02 * i, 2) for i in range(20)],
        ylabel="best top_fraction", xlabel="d_MLP", fit_xy_label=True,
        legend_framealpha=0.4,
        title="Best performing top_fraction vs hidden layer size (d_MLP)\n" + variant)


def plot_best_top_fraction_grid(path=None, figsize=(15, 11)):
    """2x2 grid like plot_best_S_grid but for best_top_fraction.

    Same panels (input_vocab_size & output_vocab_size scatters on top,
    accuracy_threshold & any_all_most strips below) and the same plain-English
    x labels, but every y axis is LINEAR with gridlines at the swept
    top_fraction values (top_fraction can be 0, which a log axis can't show);
    only the 0.04 multiples are labelled to keep the axes readable."""
    tf_kwargs = dict(path=path, y_field="best_top_fraction",
                     ylabel="best top_fraction",
                     yticks=[round(0.02 * i, 2) for i in range(20)])
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    _scatter_best_S_vs("input_vocab_size", spread=0.0087, ax=axes[0][0],
                       y_log=False, fit_xy_label=True, legend_framealpha=0.4,
                       **tf_kwargs)
    _scatter_best_S_vs("output_vocab_size", spread=0.012, ax=axes[0][1],
                       y_log=False, fit_xy_label=True, legend_framealpha=0.4,
                       **tf_kwargs)
    _strip_best_S_by("accuracy_threshold", spread=0.0197, ax=axes[1][0],
                     **tf_kwargs)
    _strip_best_S_by("any_all_most", spread=0.0296, ax=axes[1][1],
                     order=["any", "most", "all"], xlabel="", **tf_kwargs)

    # Plain-English x-axis labels ("" is the any/most/all panel, which hides its
    # label when standalone; in the grid it gets a descriptive one).
    plain_xlabels = {
        "input_vocab_size": "input vocab size",
        "output_vocab_size": "output vocab size",
        "accuracy_threshold": "required accuracy",
        "": "how many attempts must succeed",
    }
    for a in axes.ravel():  # drop the per-panel titles + enlarge text in the grid
        a.set_title("")
        a.set_xlabel(plain_xlabels.get(a.get_xlabel(), a.get_xlabel()))
        # Label only the 0.04 multiples; gridlines stay at every 0.02 tick.
        a.set_yticklabels([f"{v:.2f}" if round(v * 100) % 4 == 0 else ""
                           for v in a.get_yticks()])
        a.xaxis.label.set_size(15)
        a.yaxis.label.set_size(15)
        a.tick_params(labelsize=13)
        leg = a.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_fontsize(12)
    variant = "hand-coded model" if path is None else "hybrid model"
    fig.suptitle("Best performing top_fraction vs everything else\n" + variant,
                 fontsize=18, y=0.995)
    fig.tight_layout()
    # tight_layout ignores the suptitle; set the axes top explicitly so the
    # panels start right below the two title lines.
    fig.subplots_adjust(top=0.925)
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
y_log = True
_ = plot_capacity_vs_d(path=os.path.join(RESULTS_DIR,
                                         "capacity_search_results_topn.json"),
                       y_field="best_S", y_log = y_log)
_ = plot_capacity_vs_d(path=os.path.join(RESULTS_DIR,
                                         "capacity_search_results_topfrac.json"),
                       y_field="best_S", y_log=y_log)
_ = plot_capacity_vs_d(path=os.path.join(RESULTS_DIR,
                                         "capacity_search_results_topfrac_hybrid.json"),
                       y_field="best_S", y_log=y_log)

# %%
y_log = False
_ = plot_capacity_vs_d(path=os.path.join(RESULTS_DIR,
                                         "capacity_search_results_topn.json"),
                       y_field="best_top_n", y_log=y_log)
_ = plot_capacity_vs_d(path=os.path.join(RESULTS_DIR,
                                         "capacity_search_results_topfrac.json"),
                       y_field="best_top_fraction", y_log=y_log)
# %%
_ = plot_capacity_vs_d_and_e7()

# %%
_ = plot_capacity_vs_d_all_models()
# %%
# Less busy variant: only the 'most' rule -> one line per model and threshold.
_ = plot_capacity_vs_d_all_models(any_all_most=("most",))
# %%
# best_S vs d for the two S-sweeping variants (hand-coded and hybrid).
_ = plot_best_S_vs_d()
# %%
# best_top_fraction vs d for the same two variants (linear y axis).
_ = plot_best_top_fraction_vs_d()

# %%
_ = plot_decoupled_capacity_grids(accuracy_threshold=1.0, any_all_most="any")
# %%
_ = plot_best_S_vs_d_ff(figsize=(6, 6))
# %%
_ = plot_best_S_vs_input_vocab()
_ = plot_best_S_vs_output_vocab()
_ = plot_best_S_vs_accuracy_threshold()
_ = plot_best_S_vs_any_all_most()
# %%
_ = plot_best_S_grid()
# %%
# top_fraction versions of the d_ff scatter and the 2x2 grid (linear y axes).
_ = plot_best_top_fraction_vs_d_ff()
_ = plot_best_top_fraction_grid()
# %%
for accuracy_threshold in [0.9, 1.0]:
    for any_all_most in ["any", "most", "all"]:
        plot_decoupled_capacity_grids(accuracy_threshold, any_all_most)

# %%
_ = plot_best_S_vs_input_vocab()
# %%
_ = plot_best_S_vs_output_vocab()
# %%
_ = plot_best_S_grid()
# %%
_ = plot_capacity_vs_d_all_models(figsize=(9, 7))
# %%
# Same plot but with only the "any" data (fits use only these points too).
# Markers only, marker encodes the threshold, legend interleaves data/fit.
_ = plot_capacity_vs_d_any_only(figsize=(7, 5), error_bars=False)

# %%
_ = plot_best_S_vs_d(figsize=(8, 5))
# %%
_ = plot_best_S_vs_d_ff(figsize=(6, 5))

_ = plot_best_S_grid(figsize=(12, 8))
# %%
_ = plot_best_S_vs_d_ff(figsize=(6, 5), path=os.path.join(RESULTS_DIR, "capacity_search_results_topfrac_decoupled_hybrid.json"))

_ = plot_best_S_grid(figsize=(12, 8), path=os.path.join(RESULTS_DIR, "capacity_search_results_topfrac_decoupled_hybrid.json"))
# %%
_ = plot_best_top_fraction_vs_d(figsize=(8, 5))
# %%


_ = plot_best_top_fraction_vs_d_ff(figsize=(6, 5))

_ = plot_best_top_fraction_grid(figsize=(12, 8))

_ = plot_best_top_fraction_vs_d_ff(figsize=(6, 5),
                                   path=os.path.join(RESULTS_DIR, "capacity_search_results_topfrac_decoupled_hybrid.json"))

_ = plot_best_top_fraction_grid(figsize=(12, 8),
                                path=os.path.join(RESULTS_DIR, "capacity_search_results_topfrac_decoupled_hybrid.json"))
# %%