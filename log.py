# ── Result logging / loading ────────────────────────────────────────────────

from models import ModelSettings

import json
import os
from datetime import datetime
import copy

RESULTS_DIR = "results"

def log_result(name: str, max_facts: int, settings: ModelSettings,
               filepath: str, extra: dict | None = None) -> str:
    """Append a sub-experiment result to a human-readable log file and a companion JSON file.

    Args:
        name:      Descriptive name of the sub-experiment.
        max_facts: The capacity found by the search.
        settings:  The ModelSettings used for this run.
        filepath:  Base path (without extension) for the log files.
                   Defaults to ``results/experiment_log``.
        extra:     Any additional key/value pairs to store.

    Returns:
        The base filepath used.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if filepath is None:
        filepath = os.path.join(RESULTS_DIR, "experiment_log")

    record = {
        "name": name,
        "max_facts": max_facts,
        "timestamp": datetime.now().isoformat(),
        "settings": vars(copy.deepcopy(settings)),
    }
    if extra:
        record["extra"] = extra

    if False: #I find the .jsonl format more useful, so I'm disabling the human-readable .txt logging for now.
        # --- human-readable .txt ---
        with open(filepath + ".txt", "a", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"  Name       : {name}\n")
            f.write(f"  Max facts  : {max_facts}\n")
            f.write(f"  Timestamp  : {record['timestamp']}\n")
            f.write(f"  Settings:\n")
            for k, v in record["settings"].items():
                f.write(f"    {k:25s}: {v}\n")
            if extra:
                f.write(f"  Extra:\n")
                for k, v in extra.items():
                    f.write(f"    {k:25s}: {v}\n")
            f.write("=" * 60 + "\n\n")

    # --- machine-readable .jsonl (one JSON object per line) ---
    with open(filepath + ".jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return filepath


def load_results(filepath: str | None = None) -> list[dict]:
    """Load all logged results from the JSONL companion file.

    Args:
        filepath: Base path (without extension). Defaults to ``results/experiment_log``.

    Returns:
        A list of dicts, one per logged sub-experiment, in chronological order.
    """
    if filepath is None:
        filepath = os.path.join(RESULTS_DIR, "experiment_log")

    jsonl_path = filepath + ".jsonl"
    if not os.path.exists(jsonl_path):
        return []

    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results