# RunPod-era files (archived)

These are from an earlier plan to run the capacity sweeps on a **single RunPod
pod** with local multi-process parallelism, before the project switched to
**Modal** (`capacity_search.py` + `E*_modal.py`). They are **not used** by the
current Modal workflow and are kept here only in case the no-Modal, run-locally
approach is wanted again.

- `run_one.py` — runs ONE capacity sweep (one architecture/`d`/seed) as an
  isolated process; writes its result to `<out-dir>/parts/`.
- `launcher.py` — fans many `run_one.py` processes out across the cores of a
  single machine, then merges the `parts/` files into per-series `.jsonl`s.
- `CLOUD_SETUP.md` — the original RunPod setup guide (A100 pod, network volume).

## To use them again

Move `run_one.py` and `launcher.py` back to the **repo root** before running —
they import the project modules (`models`, `capacity_search`, `log`, `device`)
and `launcher.py` invokes `run_one.py` by relative path, both of which assume
the repo root is the working directory.

They rely on `find_max_facts(..., use_modal=False)`, so they run training
locally (CPU or a local GPU) — no Modal involved.
