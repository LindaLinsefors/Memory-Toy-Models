#%%
import torch
torch.set_default_device("cuda")

import hand_coded_models.hc2
import importlib
importlib.reload(hand_coded_models.hc2)
from hand_coded_models.hc2 import *


accuracy_threshold = 0.5
print(f'accuracy_threshold: {accuracy_threshold}')

max_facts = []
for d in [16, 32, 64, 128]:
    overall_best_n_facts, overall_best_S, overall_best_top_n, overall_best_acc = search_max_facts2(
        d=d, accuracy_threshold=accuracy_threshold, verbose=False)
    max_facts.append(overall_best_n_facts)
    print(f"d: {d}, overall_best_n_facts: {overall_best_n_facts}, overall_best_S: {overall_best_S}, overall_best_top_n: {overall_best_top_n}, overall_best_acc: {overall_best_acc}")

print (f"\nmax_facts = {max_facts}")
# %%



"""
accuracy_threshold=0.9

max_facts = [80, 256, 832, 2688]

d: 16, overall_best_n_facts: 80, overall_best_S: 4, overall_best_top_n: 2, overall_best_acc: 0.925000011920929
d: 32, overall_best_n_facts: 256, overall_best_S: 5, overall_best_top_n: 3, overall_best_acc: 0.9140625
d: 64, overall_best_n_facts: 832, overall_best_S: 6, overall_best_top_n: 8, overall_best_acc: 0.9026442766189575
d: 128, overall_best_n_facts: 2688, overall_best_S: 10, overall_best_top_n: 20, overall_best_acc: 0.9118303656578064

Second run:
max_facts = [80, 224, 832, 2688]
"""

max_facts = [80, 256, 832, 2688]
ds = [16, 32, 64, 128]

import matplotlib.pyplot as plt

plt.loglog(ds, max_facts, '-o')
plt.xlabel("d_residual")
plt.ylabel("max_facts")
ax = plt.gca()
ax.set_xticks([16, 32, 64, 128])
ax.set_xticks([], minor=True)
ax.set_xticklabels([16, 32, 64, 128])
ax.set_yticks([64, 128, 256, 512, 1024, 2048, 4096])
ax.set_yticks([], minor=True)
ax.set_yticklabels([64, 128, 256, 512, 1024, 2048, 4096])
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.show()

#Calculate the slope of the line in the log-log plot, which corresponds to the exponent in the power law relating d_residual and max_facts.
import numpy as np
log_ds = np.log(ds)
log_max_facts = np.log(max_facts)
slope = (log_max_facts[-1] - log_max_facts[0]) / (log_ds[-1] - log_ds[0])
print(f"Slope of the line in the log-log plot: {slope}")

# %%
"""
accuracy_threshold: 1
d: 16, overall_best_n_facts: 48, overall_best_S: 1, overall_best_top_n: 0, overall_best_acc: 1.0
d: 32, overall_best_n_facts: 96, overall_best_S: 5, overall_best_top_n: 1, overall_best_acc: 1.0
d: 64, overall_best_n_facts: 192, overall_best_S: 1, overall_best_top_n: 0, overall_best_acc: 1.0
d: 128, overall_best_n_facts: 512, overall_best_S: 10, overall_best_top_n: 3, overall_best_acc: 1.0

max_facts = [48, 96, 192, 512]
"""


#%%

max_facts ={0.9: [80, 256, 832, 2688], 1: [48, 96, 192, 512]}
ds = [16, 32, 64, 128]

import matplotlib.pyplot as plt

for acc_threshold, mf in max_facts.items():
    plt.loglog(ds, mf, '-o', label=f'accuracy_threshold={acc_threshold}')

plt.xlabel("d_residual")
plt.ylabel("max_facts")
ax = plt.gca()
ax.set_xticks([16, 32, 64, 128])
ax.set_xticks([], minor=True)
ax.set_xticklabels([16, 32, 64, 128])
ax.set_yticks([64, 128, 256, 512, 1024, 2048, 4096])
ax.set_yticks([], minor=True)
ax.set_yticklabels([64, 128, 256, 512, 1024, 2048, 4096])
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.legend()
plt.show()

#Calculate the slope of the line in the log-log plot, which corresponds to the exponent in the power law relating d_residual and max_facts.
import numpy as np
log_ds = np.log(ds)
for acc_threshold, mf in max_facts.items():
    log_mf = np.log(mf)
    slope = (log_mf[-1] - log_mf[0]) / (log_ds[-1] - log_ds[0])
    print(f"Slope of the line in the log-log plot for accuracy_threshold={acc_threshold}: {slope}")
# %%
