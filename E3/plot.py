#%%

import matplotlib.pyplot as plt


from log import load_results

data ={}

data['full'] = load_results("E3/full_model_log")
data['full_duplicate'] = load_results("E3/full_model_log_duplicate")
data['reduced'] = load_results("E3/reduced_model_log")
data['reduced_duplicate'] = load_results("E3/reduced_model_log_duplicate")

for key in ['full', 'full_duplicate', 'reduced', 'reduced_duplicate']:
    model_dim = [record['settings']['d_ff'] for record in data[key]]
    max_facts = [record['max_facts'] for record in data[key]]

    # ######### REMOVE THIS WHEN THE THIS EXPERIMENT IS FULLY RUN ######### #
    if key == 'full_duplicate':
        model_dim.append(128)
        max_facts.append(28672)

    print(f"\nResults for {key}:")
    print(model_dim)
    print(max_facts)
    plt.loglog(model_dim, max_facts, marker='o', label=key)

plt.xlabel("Model dimension")
plt.ylabel("Max facts")
plt.title("Log-Log Plot: Max facts vs Model dimension")
plt.legend()

xticks = model_dim
yticks = [2**i for i in range(9, 15)]

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
4
# %%
