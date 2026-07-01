# hc2_sweep_results — contents catalog

Results for the HandCodedModel2 experiments. **Only the high-level
`capacity_search_results_*` logs sit directly in this folder; every per-`n_facts`
sweep grid lives in a subfolder.**

There are two kinds of "sweep grid" file, distinguished by what each record contains:
- **top_fraction grids** — records have a `top_fraction` field. Produced by
  `hc2_sweep.py` (broad sweeps) and `hc2_capacity_search_top_fraction.py`.
- **top_n grids** — records have a `top_n` field. Produced by
  `hc2_capacity_search.py` (the current capacity search).

Each grid file is `..._d{d}_nfacts{nf}.json` with `{settings, results}`; `results`
is a flat list of per-(S, top_fraction-or-top_n, attempt) `best_guess_accuracy`
values. Repeated `(d, nf)` files get a `_(i)` suffix (never overwritten).

---

## Files directly in this folder (logs only)

| file | what it is |
|---|---|
| `capacity_search_results.json` | Capacity-search summary log, **top_fraction** search (`hc2_capacity_search_top_fraction.py`). One JSON object per line (JSONL); fields incl. `d, max_facts, best_top_fraction, best_S, precision`. (30 rows) |
| `capacity_search_results_sorted.json` | Sorted copy of the above, written by `write_sorted_capacity_results()` in `hc2_sweep_plot.py`. |
| `capacity_search_results_topn.json` | Capacity-search summary log, **top_n** search (`hc2_capacity_search.py`). JSONL; fields incl. `d, max_facts, best_top_n, best_S, precision, search_mode="top_n"`. (30 rows — from the June-19 run; coarse precision, see note.) |
| `capacity_search_results_posdown.json` | Capacity-search summary log for the **positive-down-connection** variant (`hc2_capacity_search_top_fraction.py` with `add_possitive_down_connections = True`). Same fields as `capacity_search_results.json` plus `add_possitive_down_connections`. Created only when that variant is run. |
| `capacity_search_results_topn_posdown.json` | Same idea for the **top_n** search: `hc2_capacity_search.py` with `add_possitive_down_connections = True`. Created only when that variant is run. |
| `README.md` | This file. |

---

## Subfolders (all sweep grids)

| folder | count | what / which run | used by code? |
|---|---|---|---|
| `top_fraction_grids/` | 112 | All **top_fraction** sweep grids: `hc2_sweep.py` broad sweeps (have top-level `timestamp`, `top_fraction_sweep` starts at 0.0) + `hc2_capacity_search_top_fraction.py` probes (no timestamp, `top_fraction_sweep` starts at 0.1). | **ACTIVE** — written by `hc2_sweep.py` & the top_fraction capacity search; read by `hc2_sweep_plot.py` and the top_fraction capacity search. |
| `top_fraction_grids_posdown/` | (created on run) | top_fraction grids from the **positive-down-connection** model variant (`hc2_capacity_search_top_fraction.py` with `add_possitive_down_connections = True`). Kept separate so they never pool with the default model's grids. Records' settings carry `add_possitive_down_connections: true`. | **ACTIVE** — read+written by that variant only. |
| `topn_grids/` | 0 | **Current top_n capacity grids** — scheme: one shared connection matrix per `(d,S)` + a per-attempt **column shuffle**, `precision = d/2`. Empty now; the next `hc2_capacity_search.py` run populates it. | **ACTIVE** — read+written by `hc2_capacity_search.py`. |
| `topn_grids_archive_jun19/` | 60 | top_n grids from the **June-19 run**: shared matrix but **no column shuffle**, and the old `8/32/128/512` precision (the run whose numbers came out low / 0). Records have `tie_seed`. | **ARCHIVE** — not read by current code. To reuse, move files into `topn_grids/`. |
| `topn_per_attempt_matrix/` | 36 grids + 1 log | The **earliest top_n run**: a fresh independent matrix per `(S, attempt)` (records have `conn_seed`, d≤64 only). Contains its own `capacity_search_results_topn.json` (18 rows). | **ARCHIVE** — not read by current code. |
| `duplicates/` | 14 | `_(i)` duplicate copies of `hc2_sweep.py` top_fraction grids, from running `hc2_sweep.py` several times (all `d=128`). | **ARCHIVE** — not read by loaders. |

Connection matrices are cached separately in `../conn_cache/` (one
`d{D}_s{S}.npy` per `(d,S)`), not in this folder.

---

## How to tell a grid file's origin
1. Open it; look at `results[0]`:
   - has **`top_fraction`** → a top_fraction grid.
     - payload has top-level **`timestamp`** and `settings.top_fraction_sweep[0] == 0.0` → `hc2_sweep.py`.
     - no timestamp, `top_fraction_sweep[0] == 0.1` → `hc2_capacity_search_top_fraction.py`.
   - has **`top_n`**:
     - field **`conn_seed`** → per-attempt-matrix run (archive).
     - field **`tie_seed`** → shared-matrix run (June-19 archive, or current `topn_grids/`).

## Note on the top_n results
The on-disk top_n grids (`topn_grids_archive_jun19/`) and the
`capacity_search_results_topn.json` rows predate two fixes: the per-attempt
**column shuffle** and the **`precision = d/2`** change. A fresh
`hc2_capacity_search.py` run writes corrected grids into `topn_grids/` and appends
corrected rows to `capacity_search_results_topn.json` (old rows are kept, not
overwritten).
