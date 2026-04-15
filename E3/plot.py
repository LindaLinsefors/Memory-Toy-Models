#%%

import matplotlib.pyplot as plt
import torch
import numpy as np

from log import load_results


data = load_results("E3/full_model_log")


model_dim = [record['settings']['d_ff'] for record in data]
max_facts = [record['max_facts'] for record in data]

print(model_dim)
print(max_facts)

plt.loglog(model_dim, max_facts, marker='o')
plt.xlabel("Model dimension")
plt.ylabel("Max facts")
plt.title("Log-Log Plot: Model dimension vs Max facts")

xticks = model_dim
yticks = [2**i for i in range(9, 14)]

ax = plt.gca()
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels([str(d) for d in xticks])
ax.set_yticklabels([str(d) for d in yticks])
ax.xaxis.set_minor_locator(plt.NullLocator())
ax.yaxis.set_minor_locator(plt.NullLocator())

plt.grid()
plt.show()

# %%
