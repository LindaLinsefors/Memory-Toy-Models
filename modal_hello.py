"""Throwaway Modal example — NESTED parallelism. Not part of the project.

Run it with:
    python -m modal run modal_hello.py

The shape mirrors what you want for the real sweeps:

    local_entrypoint
        └── outer.map(d in [16,32,64,128])      ← OUTER fan-out: one container per d
                └── inner.map(attempt in 0..2)  ← INNER fan-out: one container per attempt

Mapping to the real code:
    inner(d, attempt)  ~  _try_n_facts        (one independent attempt; fresh model)
    outer(d)           ~  find_max_facts       (one model size; fans its attempts out)
    main()             ~  the experiment driver (fans the model sizes out)

The "training" here is just time.sleep + a fake number so you can watch the
containers spin up. Indentation in the log prefixes shows the nesting level.
"""

import modal

app = modal.App("mtm-nested")


# ── INNER unit: one attempt ──────────────────────────────────────────────────
# This is the leaf of the tree — think "one _try_n_facts call".
@app.function()
def inner(d: int, attempt: int) -> int:
    import os
    import random
    import time

    t0 = time.time()
    time.sleep(2)  # pretend this is a training run
    # Fake result: larger d "learns" more facts, with per-attempt noise so that
    # different attempts give different answers (like real random inits do).
    rng = random.Random(d * 1000 + attempt)
    facts = d * 4 + rng.randint(0, d)
    print(f"      [inner] d={d} attempt={attempt} -> facts={facts} "
          f"(pid={os.getpid()}, {time.time() - t0:.1f}s)")
    return facts


# ── OUTER unit: one model size ───────────────────────────────────────────────
# Runs in its OWN container, and from there launches the inner fan-out.
# Think "find_max_facts for one d, parallelizing number_of_attempts".
@app.function()
def outer(d: int, n_attempts: int = 3) -> dict:
    import os

    print(f"  [outer] d={d}: launching {n_attempts} attempts in parallel "
          f"(pid={os.getpid()})")

    # THE NESTING: a Modal function calling another Modal function's .map().
    # inner.map zips the two iterables -> inner(d, 0), inner(d, 1), inner(d, 2).
    attempt_results = list(inner.map([d] * n_attempts, range(n_attempts)))

    best = max(attempt_results)  # "any/best attempt wins"
    print(f"  [outer] d={d}: attempts={attempt_results} -> best={best}")
    return {"d": d, "best": best, "attempts": attempt_results}


# ── DRIVER: runs on your laptop, fans out across model sizes ──────────────────
@app.local_entrypoint()
def main():
    ds = [16, 32, 64, 128]
    print(f"[local] launching outer() for d={ds}; each fans out its own attempts\n")

    # OUTER fan-out: one container per d. Each of those then does its own
    # INNER fan-out. So this single call triggers a 2-level tree of containers.
    for res in outer.map(ds):
        print(f"[local] result: d={res['d']:>3}  best={res['best']:>4}  "
              f"(attempts={res['attempts']})")
