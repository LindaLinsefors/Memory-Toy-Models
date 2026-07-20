#%%

import os
from collections import defaultdict

import numpy as np
from log import load_results
import matplotlib.pyplot as plt

include = ["simple", "full"]  # include only these model types in the plot; set to None to include all

#list every .jason file in E6, except for the test_log and any empty files, and remove .jasonl from the filename

experiment_dir = "E7"
log_files = [f[:-6] for f in os.listdir(experiment_dir) if f.endswith(".jsonl") and f != "test_log.jsonl" and os.path.getsize(os.path.join(experiment_dir, f)) > 0]

# Color encodes the model type; marker encodes the any/most/all rule
# (matching the capacity-vs-model-size plots in hc2_sweep_plot.py).
group_colors = {"simple": "tab:blue", "full": "mediumvioletred", "nb": "tab:gray"}
category_markers = {"any": "o", "most": "^", "all": "x"}

# Keep only groups listed in include (None = keep all)
if include is not None:
    log_files = [f for f in log_files if any(g in f for g in include)]

# Order legend: simple before full before nb, then any -> most -> all
group_order = {"simple": 0, "full": 1, "nb": 2}
category_order = {"any": 0, "most": 1, "all": 2}
def sort_key(name):
    group = next((v for k, v in group_order.items() if k in name), 99)
    category = next((v for k, v in category_order.items() if k in name), 99)
    return (group, category, name)
log_files = sorted(log_files, key=sort_key)

results = {}
group_points = defaultdict(list)  # group ("simple"/"full"/...) -> [(d, max_facts), ...]
series_data = {}  # (group, category) -> (ds, max_facts), saved for the second plot below
for log_file in log_files:
    results = load_results(os.path.join(experiment_dir, log_file))
    ds = []
    max_facts = []
    for run in results:
        ds.append(run['settings']['d_residual'])
        max_facts.append(run["max_facts"])

    group = next((g for g in group_order if g in log_file), None)
    category = next((c for c in category_order if c in log_file), None)
    series_data[(group, category)] = (ds, max_facts)

    plt.loglog(ds, max_facts, "-", marker=category_markers.get(category),
               markersize=5, color=group_colors.get(group),
               label=f"{group}, {category}")

    # Pool points by group (any/most/all together) for the power-law fits below.
    for d, mf in zip(ds, max_facts):
        if mf and mf > 0:
            group_points[group].append((d, mf))

# Fits of fixed shape max_facts = C * d^2/log(d), one per group, pooling the
# any/most/all points together. Only the coefficient C is fitted (least squares
# in log space). Broad translucent lines in the group's color.
for group in sorted(group_points, key=lambda g: group_order.get(g, 99)):
    pts = sorted(group_points[group])
    if len(pts) < 2:
        continue
    fx = np.array([d for d, _ in pts], dtype=float)
    fy = np.array([mf for _, mf in pts], dtype=float)
    # log(C) = mean(log(fy) - log(fx^2/log(fx))).
    shape = fx ** 2 / np.log(fx)
    C = np.exp(np.mean(np.log(fy) - np.log(shape)))
    # d^2/log(d) is curved in log-log space, so sample the line densely.
    xline = np.geomspace(fx.min(), fx.max(), 100)
    plt.loglog(xline, C * xline ** 2 / np.log(xline),
               color=group_colors.get(group), alpha=0.4,
               linestyle="-", linewidth=3,
               label=f"fit {group}: {C:.3g}·d²/log(d)")

plt.xlabel("model size (d)")
plt.ylabel("max facts")
plt.title("Capacity vs model size")
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

x_ticks = [16, 32, 64, 128]
y_ticks = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

ax = plt.gca()
# Remove the automatic (minor) log-scale ticks so only the ticks below remain
ax.xaxis.set_minor_locator(plt.NullLocator())
ax.yaxis.set_minor_locator(plt.NullLocator())
plt.xticks(x_ticks, x_ticks)
plt.yticks(y_ticks, y_ticks)

plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Second plot: same as above but only the "simple, any" and "full, any" series,
# with power-law fits computed from just that data.

plt.figure(figsize=(5.5, 4))

y_ticks = [512, 1024, 2048, 4096, 8192, 16384, 32768]

for group in ["simple", "full"]:
    if (group, "any") not in series_data:
        continue
    ds, max_facts = series_data[(group, "any")]
    plt.loglog(ds, max_facts, linestyle="none", marker=category_markers["any"],
               markersize=6, color=group_colors[group],
               label=f"{group}")

    pts = sorted((d, mf) for d, mf in zip(ds, max_facts) if mf and mf > 0)
    if len(pts) < 2:
        continue
    fx = np.array([d for d, _ in pts], dtype=float)
    fy = np.array([mf for _, mf in pts], dtype=float)
    k, b = np.polyfit(np.log(fx), np.log(fy), 1)
    C = np.exp(b)
    xline = np.array([fx.min(), fx.max()])
    plt.loglog(xline, C * xline ** k, color=group_colors[group], alpha=0.4,
               linestyle="-", linewidth=4,
               label=f"best fit: {C:.3g}·d^{k:.2f}")

plt.xlabel("model size (d)")
plt.ylabel("max facts")
plt.title("Capacity vs model size")
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

ax = plt.gca()
ax.xaxis.set_minor_locator(plt.NullLocator())
ax.yaxis.set_minor_locator(plt.NullLocator())
plt.xticks(x_ticks, x_ticks)
plt.yticks(y_ticks, y_ticks)

plt.grid(True)
plt.tight_layout()
plt.show()


# %%
# Third plot: same as second plot,
# but the fitted line should be best fit for
# max_facts proportional to d^2/log(d), for each of simple and full.
# The shape d^2/log(d) is fixed; only the coefficient is fitted (least squares
# in log space).

plt.figure(figsize=(5.5, 4))

y_ticks = [512, 1024, 2048, 4096, 8192, 16384, 32768]

for group in ["simple", "full"]:
    if (group, "any") not in series_data:
        continue
    ds, max_facts = series_data[(group, "any")]
    plt.loglog(ds, max_facts, linestyle="none", marker=category_markers["any"],
               markersize=6, color=group_colors[group],
               label=f"{group}")

    pts = sorted((d, mf) for d, mf in zip(ds, max_facts) if mf and mf > 0)
    if len(pts) < 2:
        continue
    fx = np.array([d for d, _ in pts], dtype=float)
    fy = np.array([mf for _, mf in pts], dtype=float)
    # Fit only the coefficient C for max_facts = C * d^2/log(d) (fixed shape).
    # Least squares in log space: log(fy) = log(C) + log(d^2/log(d)), so
    # log(C) = mean(log(fy) - log(fx^2/log(fx))).
    shape = fx ** 2 / np.log(fx)
    C = np.exp(np.mean(np.log(fy) - np.log(shape)))
    # d^2/log(d) is curved in log-log space, so sample the line densely.
    xline = np.geomspace(fx.min(), fx.max(), 100)
    plt.loglog(xline, C * xline ** 2 / np.log(xline),
               color=group_colors[group], alpha=0.4,
               linestyle="-", linewidth=4,
               label=f"best fit: {C:.3g}·d²/log(d)")

plt.xlabel("model size (d)")
plt.ylabel("max facts")
plt.title("Capacity vs model size")
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

ax = plt.gca()
ax.xaxis.set_minor_locator(plt.NullLocator())
ax.yaxis.set_minor_locator(plt.NullLocator())
plt.xticks(x_ticks, x_ticks)
plt.yticks(y_ticks, y_ticks)

plt.grid(True)
plt.tight_layout()
plt.show()
# %%
