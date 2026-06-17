#%%

import os
from log import load_results
import matplotlib.pyplot as plt

include = ["reduced", "full"]  # include only these model types in the plot; set to None to include all

#list every .jason file in E6, except for the test_log and any empty files, and remove .jasonl from the filename    

experiment_dir = "E6"
log_files = [f[:-6] for f in os.listdir(experiment_dir) if f.endswith(".jsonl") and f != "test_log.jsonl" and os.path.getsize(os.path.join(experiment_dir, f)) > 0]

colors = {"all": "C1", "any": "C2", "most": "C0"}
styles = {"full": "-o", "reduced": "--x", "nb": ":s"}

# Keep only groups listed in include (None = keep all)
if include is not None:
    log_files = [f for f in log_files if any(g in f for g in include)]

# Order legend: reduced before full before nb, then any -> most -> all
group_order = {"reduced": 0, "full": 1, "nb": 2}
category_order = {"any": 0, "most": 1, "all": 2}
def sort_key(name):
    group = next((v for k, v in group_order.items() if k in name), 99)
    category = next((v for k, v in category_order.items() if k in name), 99)
    return (group, category, name)
log_files = sorted(log_files, key=sort_key)

results = {}
for log_file in log_files:
    results = load_results(os.path.join(experiment_dir, log_file))
    ds = []
    max_facts = []
    for run in results:
        ds.append(run['settings']['d_residual'])
        max_facts.append(run["max_facts"])

    color = next((c for k, c in colors.items() if k in log_file), None)
    style = next((s for k, s in styles.items() if k in log_file), "-o")

    plt.loglog(ds, max_facts, style, color=color, label=log_file)

plt.xlabel("d_residual")
plt.ylabel("max_facts")
plt.title("E6: max_facts vs d_residual")
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

x_ticks = [16, 32, 64, 128]
y_ticks = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

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
