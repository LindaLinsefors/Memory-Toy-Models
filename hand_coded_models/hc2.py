#%%
"""
Generate binary connection matrices between D neurons and T features.

Each column has exactly S ones (each feature connects to S neurons).
Row sums are balanced (~S*T/D per neuron).
Any two columns share at most 1 row (ideally).

The "at most 1 shared neuron" property is a combinatorial design constraint.
It is achievable only when T <= D*(D-1) / (S*(S-1)). Beyond that bound,
some feature pairs must share >= 2 neurons; the algorithm minimises violations.
"""

import numpy as np
from itertools import combinations
from typing import Optional
import warnings


def make_connection_matrix(
    D: int,
    T: int,
    S: int,
    seed: Optional[int] = None,
    n_restarts: int = 5,
    sa_steps: int = 30_000,
) -> np.ndarray:
    """
    Generate a D x T binary connection matrix.

    Args:
        D: number of neurons (rows)
        T: number of features (columns)
        S: neurons per feature -- each column has exactly S ones
        seed: random seed for reproducibility
        n_restarts: independent attempts; returns the best result
        sa_steps: simulated-annealing steps per attempt

    Returns:
        numpy int8 array of shape (D, T)
    """
    if not (0 < S <= D):
        raise ValueError(f"Need 0 < S <= D, got S={S}, D={D}")
    if D <= 0 or T <= 0:
        raise ValueError("D and T must be positive integers")

    if S > 1:
        max_feasible_T = D * (D - 1) // (S * (S - 1))
        if T > max_feasible_T:
            warnings.warn(
                f"T={T} exceeds the theoretical maximum ({max_feasible_T}) for "
                f"pairwise overlap <= 1 with D={D}, S={S}. "
                f"Some feature pairs will inevitably share >= 2 neurons.",
                stacklevel=2,
            )

    rng = np.random.default_rng(seed)
    best_matrix = None
    best_score = float("inf")

    for _ in range(n_restarts):
        m = _greedy_init(D, T, S, rng)
        m = _sa_improve(m, D, T, S, rng, steps=sa_steps)
        m = _balance_fix(m, T)  # greedy balance pass without new violations
        score = _total_score(m)
        if score < best_score:
            best_score = score
            best_matrix = m.copy()

    return best_matrix


def evaluate_connection_matrix(M: np.ndarray) -> dict:
    """
    Compute quality statistics for a connection matrix.

    Returns a dict with column sums (should all equal S), row sum statistics,
    max pairwise overlap between any two feature columns, and overlap violation counts.
    """
    D, T = M.shape
    row_sums = M.sum(axis=1)
    col_sums = M.sum(axis=0)

    pairwise = [int(M[:, j1] @ M[:, j2]) for j1, j2 in combinations(range(T), 2)]

    return {
        "shape": (D, T),
        "col_sum_min": int(col_sums.min()),
        "col_sum_max": int(col_sums.max()),
        "row_sum_min": int(row_sums.min()),
        "row_sum_max": int(row_sums.max()),
        "row_sum_mean": float(row_sums.mean()),
        "row_sum_std": float(row_sums.std()),
        "max_pairwise_overlap": max(pairwise) if pairwise else 0,
        "overlap_violation_pairs": sum(1 for o in pairwise if o > 1),
        "overlap_violations_total": sum(o - 1 for o in pairwise if o > 1),
    }


# ── Internal helpers ──────────────────────────────────────────────────────────


def _greedy_init(D: int, T: int, S: int, rng: np.random.Generator) -> np.ndarray:
    """
    Column-by-column greedy construction.

    For each column, picks S rows by minimising:
      - current row sum (favours balance)
      - co-occurrence count with already-selected rows in prior columns (avoids overlap)
    """
    matrix = np.zeros((D, T), dtype=np.int8)
    row_sums = np.zeros(D, dtype=np.int32)
    # pair_count[i, k]: number of existing columns that contain both row i and row k
    pair_count = np.zeros((D, D), dtype=np.int32)

    for col in range(T):
        available = np.ones(D, dtype=bool)
        selected: list[int] = []

        for _ in range(S):
            cands = np.where(available)[0]
            scores = row_sums[cands].astype(np.float64)
            for sel in selected:
                # Heavy penalty for pairs that already co-appear: adding this row
                # would create a column overlap >= 2 with some earlier column.
                scores += pair_count[sel, cands] * 1_000.0
            scores += rng.uniform(0.0, 0.01, len(cands))  # break ties randomly

            pick = int(cands[np.argmin(scores)])
            selected.append(pick)
            available[pick] = False

        for i in selected:
            matrix[i, col] = 1
            row_sums[i] += 1
        for i, k in combinations(selected, 2):
            pair_count[i, k] += 1
            pair_count[k, i] += 1

    return matrix


def _sa_improve(
    matrix: np.ndarray,
    D: int,
    T: int,
    S: int,
    rng: np.random.Generator,
    steps: int,
) -> np.ndarray:
    """
    Simulated-annealing refinement.

    Each step picks a random column and swaps one 1 with one 0 in that column,
    keeping column sums exactly fixed. Accepts moves that improve the objective
    or occasionally accepts worse moves (at high temperature) to escape local optima.

    Objective = overlap_violations * 1000 + variance(row_sums).
    """
    m = matrix.copy().astype(np.int8)
    row_sums = m.sum(axis=1).astype(np.int32)

    # ov[j1, j2] = number of rows shared between columns j1 and j2; diagonal is 0
    ov = (m.T.astype(np.int32) @ m.astype(np.int32))
    np.fill_diagonal(ov, 0)

    T0, Tf = 5_000.0, 0.5
    decay = (Tf / T0) ** (1.0 / max(steps, 1))
    temp = T0

    for _ in range(steps):
        col = int(rng.integers(T))
        ones = np.where(m[:, col] == 1)[0]
        zeros = np.where(m[:, col] == 0)[0]
        if not len(ones) or not len(zeros):
            temp *= decay
            continue

        i_out = int(rng.choice(ones))
        i_in = int(rng.choice(zeros))

        # Swapping i_out -> i_in in column `col` changes ov[col, j] by:
        #   m[i_in, j] - m[i_out, j]   for each j != col
        dov = m[i_in, :].astype(np.int32) - m[i_out, :].astype(np.int32)
        dov[col] = 0  # self-overlap stays 0

        new_ov_col = ov[col, :] + dov
        viol_old = int(np.sum(np.maximum(0, ov[col, :] - 1)))
        viol_new = int(np.sum(np.maximum(0, new_ov_col - 1)))
        overlap_delta = viol_new - viol_old

        rs = row_sums.copy()
        rs[i_out] -= 1
        rs[i_in] += 1
        balance_delta = float(np.var(rs)) - float(np.var(row_sums))

        delta = overlap_delta * 1_000.0 + balance_delta

        if delta <= 0.0 or rng.random() < np.exp(-delta / temp):
            m[i_out, col] = 0
            m[i_in, col] = 1
            row_sums = rs
            ov[col, :] = new_ov_col
            ov[:, col] = new_ov_col

        temp *= decay

    return m


def _balance_fix(matrix: np.ndarray, T: int, max_iters: int = 1000) -> np.ndarray:
    """
    Greedy balance pass: repeatedly swap a 1 from a high-degree neuron to a
    low-degree neuron, accepting only moves that do not increase overlap violations.

    Stops when no improving swap exists (converged) or max_iters is reached.
    """
    m = matrix.copy()
    row_sums = m.sum(axis=1).astype(np.int32)
    ov = (m.T.astype(np.int32) @ m.astype(np.int32))
    np.fill_diagonal(ov, 0)

    for _ in range(max_iters):
        best_gain = 1  # require gain >= 2 to guarantee variance decreases
        best_swap = None

        for col in range(T):
            ones = np.where(m[:, col] == 1)[0]
            zeros = np.where(m[:, col] == 0)[0]
            if not len(ones) or not len(zeros):
                continue

            # Highest-sum rows that are currently 1, lowest-sum rows that are 0
            top_ones = ones[np.argsort(row_sums[ones])[::-1]][:3]
            bot_zeros = zeros[np.argsort(row_sums[zeros])][:3]

            for i_out in top_ones:
                for i_in in bot_zeros:
                    gain = int(row_sums[i_out]) - int(row_sums[i_in])
                    if gain <= best_gain:
                        continue
                    dov = m[i_in, :].astype(np.int32) - m[i_out, :].astype(np.int32)
                    dov[col] = 0
                    new_ov_col = ov[col, :] + dov
                    viol_delta = (int(np.sum(np.maximum(0, new_ov_col - 1))) -
                                  int(np.sum(np.maximum(0, ov[col, :] - 1))))
                    if viol_delta <= 0:
                        best_gain = gain
                        best_swap = (col, int(i_out), int(i_in), new_ov_col.copy())

        if best_swap is None:
            break

        col, i_out, i_in, new_ov_col = best_swap
        m[i_out, col] = 0
        m[i_in, col] = 1
        row_sums[i_out] -= 1
        row_sums[i_in] += 1
        ov[col, :] = new_ov_col
        ov[:, col] = new_ov_col

    return m


def _total_score(m: np.ndarray) -> float:
    row_sums = m.sum(axis=1).astype(np.int32)
    ov = (m.T.astype(np.int32) @ m.astype(np.int32))
    np.fill_diagonal(ov, 0)
    violations = int(np.sum(np.maximum(0, ov - 1))) // 2
    return violations * 1_000.0 + float(np.var(row_sums))


# ── HandCodedModel2 ───────────────────────────────────────────────────────────
# A generalised hand-coded MLP that works for any (hidden_dim, n_labels, n_facts)
# and uses make_connection_matrix instead of hard-coded lookup tables.

import torch
import torch.nn.functional as F
from models import generate_facts


class HandCodedModel2Settings:
    """Settings for HandCodedModel2.

    Differences from HandCodedModelSettings in hc_models.py:
    - n_facts may be any positive integer, not just a multiple of output_vocab_size.
    - No 'adjustments' flag (that post-hoc step did not consistently help).
    - Fixed spelling: n_neurons_per_label (was n_neruons_per_label).
    """

    def __init__(
        self,
        input_vocab_size: int = 32,
        output_vocab_size: int = 16,
        n_facts: int = 16,
        seed: int = 42,
        d_ff: int = 16,
        n_neurons_per_label: int = 3,
        use_top_no_top_fraction: str = 'top_n',
        top_n: int = 0,
        top_fraction: float = 0.2,
        add_possitive_down_connections = False
    ):
        self.seq_len = 2                   # each fact has exactly two input tokens (fixed)
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.n_facts = n_facts
        self.seed = seed
        self.d_ff = d_ff
        self.n_neurons_per_label = n_neurons_per_label
        self.use_top_no_top_fraction = use_top_no_top_fraction  # 'top_n' or 'top_fraction'
        self.top_n = top_n
        self.top_fraction = top_fraction
        self.add_possitive_down_connections = add_possitive_down_connections


class HandCodedModel2:
    """Analytically constructed two-layer MLP for memorising (input, label) facts.

    Architecture (forward pass):
        x_enc  = [one_hot(x[:,0]), one_hot(x[:,1])]   # (batch, n_vocab*2)
        hidden = relu(x_enc @ up_matrix.T)             # (batch, hidden_dim)
        logits = hidden @ down_matrix.T + down_bias    # (batch, n_labels)

    How the weights encode the facts:
    - Each output label l is assigned S "guard" neurons via make_connection_matrix.
    - A guard neuron for label l must fire only when the input does NOT belong to l,
      so its firing drives logit[l] from +1 (the bias) to a negative value.
    - Logit[l] is thus positive (predicting l) only when none of l's guard neurons fire,
      which happens exactly when the input is one of l's own facts.
    - The up_matrix starts at 1 everywhere (all neurons fire on all inputs).
      Per-neuron, token weights are lowered to -1 (for the most common tokens among
      guarded inputs) or 0 (for the remaining ones) to silence the neuron on those inputs.

    Improvements over HandCodedModel in hc_models.py:
    1. Any n_facts is valid. generate_facts assigns labels as arange(n_facts) % output_vocab_size,
       so the number of facts per label can differ by at most 1.
    2. Uses make_connection_matrix for arbitrary (hidden_dim, n_labels, S) instead of a
       small set of hard-coded assignment tables.
    3. No post-hoc adjustment step.
    """

    def __init__(self, settings: HandCodedModel2Settings, precomputed_conn: Optional[np.ndarray] = None):
        self.settings = settings

        # generate_facts supports any n_facts; labels = arange(n_facts) % output_vocab_size
        self.facts = generate_facts(
            n_facts=settings.n_facts,
            input_len=2,
            input_vocab_size=settings.input_vocab_size,
            output_vocab_size=settings.output_vocab_size,
            seed=settings.seed,
        )

        inputs = self.facts['inputs']   # (n_facts, 2) int tensor
        labels = self.facts['targets']  # (n_facts,)   int tensor
        device = labels.device          # stay on whatever device generate_facts used

        S          = settings.n_neurons_per_label
        n_labels   = settings.output_vocab_size
        hidden_dim = settings.d_ff
        n_vocab    = settings.input_vocab_size

        # --- Connection matrix: shape (hidden_dim, n_labels) ---
        # conn[neuron, label] = 1  ↔  neuron guards label.
        # make_connection_matrix guarantees exactly S ones per column (label), balanced
        # row sums, and minimal pairwise overlap between columns.
        # Use a pre-built matrix if supplied (avoids re-running SA inside search loops).
        if precomputed_conn is not None:
            conn_np = precomputed_conn
        else:
            conn_np = make_connection_matrix(D=hidden_dim, T=n_labels, S=S, seed=settings.seed)
        conn = torch.tensor(conn_np, dtype=torch.float32, device=device)  # (hidden_dim, n_labels)

        # --- Up matrix: shape (hidden_dim, n_vocab * 2) ---
        # Weights from the flattened one-hot input [first_token | second_token] to each neuron.
        # Initial value 1: relu(1 + 1) = 2, so every neuron fires on every input by default.
        # We will selectively lower weights so each neuron is silent on its guarded inputs.
        mlp_up = torch.ones(hidden_dim, n_vocab * 2, device=device)

        for neuron in range(hidden_dim):
            # Identify which facts this neuron must stay silent on.
            # conn[neuron] is a (n_labels,) vector of 0/1; True positions are guarded labels.
            neuron_guards  = conn[neuron].bool()    # (n_labels,) bool
            guarded_mask   = neuron_guards[labels]  # (n_facts,) bool — True for guarded facts
            guarded_inputs = inputs[guarded_mask]   # (k, 2) — neuron must NOT fire for these

            if guarded_inputs.shape[0] == 0:
                # No current facts fall under this neuron's guarded labels; leave at 1.
                continue

            # Count how often each token appears at each position among guarded inputs.
            # High-frequency tokens are good candidates for suppression: setting one token to -1
            # silences the neuron on every guarded input that contains it (relu(-1+1)=0).
            unique_f, counts_f = torch.unique(guarded_inputs[:, 0], return_counts=True)
            unique_s, counts_s = torch.unique(guarded_inputs[:, 1], return_counts=True)

            # Shuffle before ranking so that ties among equal-count tokens are broken randomly.
            perm_f = torch.randperm(unique_f.shape[0], device=device)
            unique_f, counts_f = unique_f[perm_f], counts_f[perm_f]
            perm_s = torch.randperm(unique_s.shape[0], device=device)
            unique_s, counts_s = unique_s[perm_s], counts_s[perm_s]

            # How many top tokens to suppress per position.
            if settings.use_top_no_top_fraction == 'top_fraction':
                k_f = max(1, int(len(unique_f) * settings.top_fraction))
                k_s = max(1, int(len(unique_s) * settings.top_fraction))
            else:  # 'top_n'
                k_f = settings.top_n
                k_s = settings.top_n

            # Top tokens: weight → -1.  relu(-1 + 1) = 0 whenever that token appears.
            top_f = unique_f[torch.argsort(counts_f, descending=True)[:k_f]]
            top_s = unique_s[torch.argsort(counts_s, descending=True)[:k_s]]
            if top_f.numel() > 0:
                mlp_up[neuron, top_f] = -1
            if top_s.numel() > 0:
                mlp_up[neuron, top_s + n_vocab] = -1

            # Remaining guarded inputs: not covered by any -1 token in position 0 OR position 1.
            # Without further action these inputs still activate the neuron (relu(1+1)=2).
            # Zero both weights so relu(0+0)=0 — neuron is silenced.
            if top_f.numel() > 0:
                covered_f = (guarded_inputs[:, 0].unsqueeze(1) == top_f.unsqueeze(0)).any(1)
            else:
                covered_f = torch.zeros(guarded_inputs.shape[0], dtype=torch.bool, device=device)

            if top_s.numel() > 0:
                covered_s = (guarded_inputs[:, 1].unsqueeze(1) == top_s.unsqueeze(0)).any(1)
            else:
                covered_s = torch.zeros(guarded_inputs.shape[0], dtype=torch.bool, device=device)

            # An input is "remaining" only if it is uncovered in BOTH positions.
            remaining = guarded_inputs[~covered_f & ~covered_s]
            if remaining.shape[0] > 0:
                mlp_up[neuron, remaining[:, 0]] = 0
                mlp_up[neuron, remaining[:, 1] + n_vocab] = 0

        # --- Down matrix: shape (n_labels, hidden_dim) ---
        # down[l, n] = -2 if neuron n guards label l, else 0.
        #
        # logit[b, l] = sum_n (hidden[b,n] * down[l,n]) + bias[l]
        #             = -2 * (sum of firing guard-neurons for l) + 1
        #
        # If a guard neuron fires (wrong input for l): logit[l] ≤ -1  → correctly negative.
        # If NO guard neuron fires (input IS a fact for l): logit[l] = +1 → correctly positive.
        self.up_matrix   = mlp_up
        if settings.add_possitive_down_connections:
            self.down_matrix = -2.0 * hidden_dim * conn.T + 1.0 * (1-conn.T)
        else:
            self.down_matrix = -2.0 * conn.T                      # (n_labels, hidden_dim)
        self.down_bias   = torch.ones(n_labels, device=device)

    def forward(self, x):
        """Run the model on input pairs x of shape (batch, 2).

        Returns (logits, hidden) — both float tensors on the same device as x.
        """
        first  = F.one_hot(x[:, 0], num_classes=self.settings.input_vocab_size).float()
        second = F.one_hot(x[:, 1], num_classes=self.settings.input_vocab_size).float()
        x_enc  = torch.cat([first, second], dim=-1)               # (batch, n_vocab*2)
        hidden = torch.relu(x_enc @ self.up_matrix.T)             # (batch, hidden_dim)
        logits = hidden @ self.down_matrix.T + self.down_bias     # (batch, n_labels)
        return logits, hidden

    def evaluate(self):
        """Evaluate the model on all stored facts.

        Returns:
            accuracy:            fraction of (fact, label) pairs where logit sign is correct
            best_guess_accuracy: fraction of facts where argmax label matches target
            logits:              (n_facts, n_labels) float tensor
            hidden:              (n_facts, hidden_dim) float tensor
        """
        logits, hidden = self.forward(self.facts['inputs'])
        one_hot_targets = F.one_hot(self.facts['targets'], self.settings.output_vocab_size)
        # Correct sign means: true-label logit > 0, all other logits < 0
        accuracy = (one_hot_targets.bool() == (logits > 0)).float().mean().item()
        best_guess_accuracy = self.facts['targets'].eq(logits.argmax(dim=1)).float().mean().item()
        return accuracy, best_guess_accuracy, logits, hidden






# ── Connection-matrix cache ───────────────────────────────────────────────────
# Building a connection matrix runs simulated annealing, which can be slow for
# large D or S.  The matrix depends only on (D, T, S, seed) — not on n_facts or
# top_n — so we cache it here and reuse it across all evaluations that share the
# same model shape.
_conn_cache: dict = {}


def _get_conn_matrix(D: int, T: int, S: int, seed: int) -> np.ndarray:
    """Return a cached connection matrix, building it on first call."""
    key = (D, T, S, seed)
    if key not in _conn_cache:
        # Suppress the "T exceeds theoretical maximum" warning — we know overlap
        # violations may occur and the SA minimises them as best it can.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _conn_cache[key] = make_connection_matrix(D=D, T=T, S=S, seed=seed)
    return _conn_cache[key]


# ── Search functions for HandCodedModel2 ─────────────────────────────────────

def search_best_top_n2(
    d: int,
    n_facts: int,
    S: int,
    retries: int = 2,
    metric: str = 'best_guess_accuracy',
    precomputed_conn: Optional[np.ndarray] = None,
    seed: int = 42,
    verbose: bool = True,
) -> tuple:
    """Search over top_n for HandCodedModel2 with fixed n_neurons_per_label S.

    Mirrors search_best_top_n in hc_models.py, but operates on HandCodedModel2
    and takes an extra S argument.

    For each top_n (starting at 0 and incrementing), the model is built `retries`
    times — torch.randperm inside the constructor provides different random
    tie-breaking on each call — and the best score is kept.  The loop stops once
    accuracy reaches 1.0 or after two consecutive drops in accuracy.

    precomputed_conn: a (D × D) numpy array from _get_conn_matrix(d, d, S, seed).
        If None, it is fetched/built automatically.  Passing it in avoids
        rebuilding the expensive SA matrix on every call.

    Returns (best_top_n, best_accuracy).
    """
    if metric not in ('accuracy', 'best_guess_accuracy'):
        raise ValueError("metric must be 'accuracy' or 'best_guess_accuracy'")

    # Build settings once and mutate only top_n inside the loop.
    settings = HandCodedModel2Settings(
        input_vocab_size=2 * d,
        output_vocab_size=d,
        n_facts=n_facts,
        seed=seed,
        d_ff=d,
        n_neurons_per_label=S,
        use_top_no_top_fraction='top_n',
        top_n=0,
    )

    # Ensure the connection matrix is available — fetched from cache if possible.
    if precomputed_conn is None:
        precomputed_conn = _get_conn_matrix(d, d, S, seed)

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
            acc, bga, _, _ = model.evaluate()
            score = acc if metric == 'accuracy' else bga
            accuracy_for_top_n = max(accuracy_for_top_n, score)
            if accuracy_for_top_n == 1.0:
                break  # perfect score — no need for more retries

        if verbose:
            print(f"    top_n={top_n}: {metric}={accuracy_for_top_n:.4f}")

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
    metric: str = 'best_guess_accuracy',
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
        conn = _get_conn_matrix(d, d, S, seed)

        def passes(k, conn=conn):
            """Return (ok, best_top_n, best_acc) for n_facts = k * d."""
            best_top_n_k, acc = search_best_top_n2(
                d, k * d, S,
                retries=retries, metric=metric,
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
                print(f"  [S={S}] n_facts={hi * d}: {metric}={acc:.4f}, best_top_n={top_n} {'✓' if ok else '✗'}")
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
                print(f"  [S={S}] n_facts={mid * d}: {metric}={acc:.4f}, best_top_n={top_n} {'✓' if ok else '✗'}")
            if ok:
                lo        = mid
                lo_result = (top_n, acc)
            else:
                hi = mid

        max_facts_for_S          = lo * d
        best_top_n_for_S, best_acc_for_S = lo_result

        if verbose:
            print(f"  [S={S}] max_facts={max_facts_for_S}, best_top_n={best_top_n_for_S}, {metric}={best_acc_for_S:.4f}")

        if max_facts_for_S > overall_best_n_facts:
            overall_best_n_facts = max_facts_for_S
            overall_best_S       = S
            overall_best_top_n   = best_top_n_for_S
            overall_best_acc     = best_acc_for_S

    if verbose:
        print(f"\nBest overall: n_facts={overall_best_n_facts}, S={overall_best_S}, "
              f"top_n={overall_best_top_n}, {metric}={overall_best_acc:.4f}")
    return overall_best_n_facts, overall_best_S, overall_best_top_n, overall_best_acc


# %%
