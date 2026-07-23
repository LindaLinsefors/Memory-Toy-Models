#%%
import torch
torch.set_default_device("cuda")

import hand_coded_models.hc2
import importlib
importlib.reload(hand_coded_models.hc2)
from hand_coded_models.hc2 import *

from typing import Optional
import numpy as np


# ── Hyperparameter sweep (measuring capacity) ─────────────────────────────────
# To measure how many facts the construction can store, sweep over its two
# hyperparameters: S (n_neurons_per_label) and top_n. search_best_top_n2 finds
# the best top_n for a fixed S; search_max_facts2 wraps it to find, over all S,
# the largest fact set the model can store above an accuracy threshold.

def search_best_top_n2(
    d: int,
    n_facts: int,
    S: int,
    retries: int = 2,
    precomputed_conn: Optional[np.ndarray] = None,
    seed: int = 42,
    verbose: bool = True,
) -> tuple:
    """Search over top_n for a fixed n_neurons_per_label S.

    For each top_n (starting at 0 and incrementing), the model is built `retries`
    times — torch.randperm inside the constructor provides different random
    tie-breaking on each call — and the best score is kept.  The loop stops once
    accuracy reaches 1.0 or after two consecutive drops in accuracy.

    precomputed_conn: a (D × D) numpy array from get_conn_matrix(d, d, S, seed).
        If None, it is fetched/built automatically.  Passing it in avoids
        rebuilding the expensive SA matrix on every call.

    Returns (best_top_n, best_accuracy).
    """
    # Build settings once and mutate only top_n inside the loop.
    settings = HandCodedModel2Settings(
        input_vocab_size=2 * d,
        output_vocab_size=d,
        n_facts=n_facts,
        seed=seed,
        d_ff=d,
        n_neurons_per_label=S,
        use_top_n_or_top_fraction='top_n',
        top_n=0,
    )

    # Ensure the connection matrix is available — fetched from cache if possible.
    if precomputed_conn is None:
        precomputed_conn = get_conn_matrix(d, d, S, seed)

    best_top_n         = 0
    best_accuracy      = 0.0
    prev_accuracy      = -1.0
    decreases_in_a_row = 0
    top_n              = 0

    while True:
        settings.top_n = top_n

        # Try several random initialisations; keep the best score for this top_n.
        accuracy_for_top_n = 0.0
        for _ in range(retries):
            model = HandCodedModel2(settings, precomputed_conn=precomputed_conn)
            accuracy, _, _ = model.evaluate()
            accuracy_for_top_n = max(accuracy_for_top_n, accuracy)
            if accuracy_for_top_n == 1.0:
                break  # perfect score — no need for more retries

        if verbose:
            print(f"    top_n={top_n}: accuracy={accuracy_for_top_n:.4f}")

        if accuracy_for_top_n > best_accuracy:
            best_accuracy = accuracy_for_top_n
            best_top_n    = top_n

        if best_accuracy == 1.0:
            break  # can't improve further

        if accuracy_for_top_n < prev_accuracy:
            decreases_in_a_row += 1
            if decreases_in_a_row >= 2:
                break  # two consecutive drops → further increases likely unhelpful
        else:
            decreases_in_a_row = 0

        prev_accuracy = accuracy_for_top_n
        top_n += 1

    return best_top_n, best_accuracy


def search_max_facts2(
    d: int,
    accuracy_threshold: float,
    retries: int = 2,
    max_S: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> tuple:
    """Find the largest n_facts that HandCodedModel2 can store above a threshold.

    Searches over both n_neurons_per_label (S = 1 … max_S) and top_n.  For
    each S value:
      1. The connection matrix is built once (SA is expensive) and cached.
      2. An exponential search finds a passing multiplier lo and a failing
         multiplier hi (n_facts = k * d, k = lo or hi).
      3. Binary search narrows the boundary between lo and hi.
      4. search_best_top_n2 is called at each candidate k to find the optimal
         top_n for that (n_facts, S) combination.

    n_facts is kept as a multiple of d (= output_vocab_size) so that label
    frequencies are perfectly balanced.

    Speed notes
    -----------
    - Connection matrices are cached in _conn_cache: the SA step runs at most
      once per (d, S) across the entire Python session.
    - Passing precomputed_conn into each model construction means the SA step
      is not re-run inside search_best_top_n2.
    - Early exits (accuracy == 1.0, two consecutive top_n drops) minimise the
      number of model builds.

    Returns (best_n_facts, best_S, best_top_n, best_accuracy), or
    (0, None, None, None) if even n_facts=d fails for every S tried.
    """
    if max_S is None:
        # Values above ~10 rarely help and S must be <= d.
        max_S = min(10, d)

    overall_best_n_facts = 0
    overall_best_S       = None
    overall_best_top_n   = None
    overall_best_acc     = None

    for S in range(1, max_S + 1):
        if verbose:
            print(f"\n[S={S}] Searching with n_neurons_per_label={S}")

        # Build (and cache) the connection matrix for this S once.
        # All n_facts / top_n evaluations below will reuse the same matrix.
        conn = get_conn_matrix(d, d, S, seed)

        def passes(k, conn=conn):
            """Return (ok, best_top_n, best_acc) for n_facts = k * d."""
            best_top_n_k, acc = search_best_top_n2(
                d, k * d, S,
                retries=retries,
                precomputed_conn=conn, seed=seed,
                verbose=verbose,
            )
            return acc >= accuracy_threshold, best_top_n_k, acc

        # --- Phase 1: exponential growth to bracket the failure point ---
        # Grow k as 1, 2, 4, 8, … until the model fails the threshold.
        # After the loop: lo passes, hi fails.
        lo        = 0
        lo_result = (None, None)
        hi        = 1

        while True:
            if verbose:
                print(f"  [S={S}] Trying n_facts={hi * d} ...")
            ok, top_n, acc = passes(hi)
            if verbose:
                print(f"  [S={S}] n_facts={hi * d}: accuracy={acc:.4f}, best_top_n={top_n} {'✓' if ok else '✗'}")
            if not ok:
                break
            lo        = hi
            lo_result = (top_n, acc)
            hi       *= 2

        if lo == 0:
            # Even n_facts = d (k=1) fails for this S — try next S value.
            if verbose:
                print(f"  [S={S}] Failed at n_facts={d}, skipping.")
            continue

        # --- Phase 2: binary search between lo (passes) and hi (fails) ---
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if verbose:
                print(f"  [S={S}] Trying n_facts={mid * d} (binary search) ...")
            ok, top_n, acc = passes(mid)
            if verbose:
                print(f"  [S={S}] n_facts={mid * d}: accuracy={acc:.4f}, best_top_n={top_n} {'✓' if ok else '✗'}")
            if ok:
                lo        = mid
                lo_result = (top_n, acc)
            else:
                hi = mid

        max_facts_for_S          = lo * d
        best_top_n_for_S, best_acc_for_S = lo_result

        if verbose:
            print(f"  [S={S}] max_facts={max_facts_for_S}, best_top_n={best_top_n_for_S}, accuracy={best_acc_for_S:.4f}")

        if max_facts_for_S > overall_best_n_facts:
            overall_best_n_facts = max_facts_for_S
            overall_best_S       = S
            overall_best_top_n   = best_top_n_for_S
            overall_best_acc     = best_acc_for_S

    if verbose:
        print(f"\nBest overall: n_facts={overall_best_n_facts}, S={overall_best_S}, "
              f"top_n={overall_best_top_n}, accuracy={overall_best_acc:.4f}")
    return overall_best_n_facts, overall_best_S, overall_best_top_n, overall_best_acc


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
