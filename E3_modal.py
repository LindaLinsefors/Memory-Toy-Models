"""E3 capacity-scaling experiment, running on Modal (GPU).

Run it with:
    python -m modal run E3_modal.py

How it parallelises (two levels):
    main()  (your laptop)
      └── find_max_facts_modal.map(over d)      ← one CPU container per model size
              └── _try_n_facts_modal.map(attempts)  ← GPU containers, the training

Results come back to your laptop, which writes the .jsonl logs locally (so the
logs end up on your machine, not on an ephemeral container).

NOTE on W&B: training runs inside Modal containers, which have no W&B
credentials, so log_to_wandb must be False here unless you set up a Modal
Secret. See the note at the bottom of this file.
"""

import copy
import os

# Importing this registers the Modal app + functions (and builds the image spec).
from capacity_search import app, find_max_facts_modal
from models import ModelSettings
from log import log_result


# ── Experiment configuration ─────────────────────────────────────────────────
experiment_dir = "E3"
testing = True                  # small/cheap run to validate the pipeline first
model_type = "reduced"          # 'full' or 'reduced'
duplicate = True

log_to_wandb = True
lr = [1e-2, 3e-3, 1e-3]
patience = 5000
n_epochs = 50000
target_accuracy = "accuracy"
threshold_to_continue = 0.99
number_of_attempts = 11
loss_type = "CE"                # 'BCE' or 'CE'
device = "gpu"                  # "gpu" or "cpu" (where the training containers run)

d_values = [16, 32, 64, 128]

if loss_type == "CE":
    target_accuracy = "best_guess_accuracy"
    threshold_to_continue = 0.95

if testing:
    lr = [1e-2]
    patience = 10
    n_epochs = 100


# Base architecture for this experiment.
if model_type == "reduced":
    base_settings = ModelSettings(attention=False, ff=True, bias=True,
                                  norms=False, ff_residual=False,
                                  ff_activation_type="ReLU")
elif model_type == "full":
    base_settings = ModelSettings(attention=True, ff=True, bias=True,
                                  norms=True, ff_residual=True,
                                  ff_activation_type="GELU")
else:
    raise ValueError("Invalid model type. Choose 'full' or 'reduced'.")


def build_log_path() -> str:
    if testing:
        return os.path.join(experiment_dir, "test_log")
    path = os.path.join(experiment_dir, f"{model_type}_model_log")
    if loss_type == "CE":
        path += "_CE"
    if duplicate:
        path += "_duplicate"
    # Make the log path unique to avoid mixing with previous runs.
    i = 1
    while os.path.exists(f"{path}_({i}).jsonl"):
        i += 1
    return f"{path}_({i})"


def build_configs():
    """One config dict per model size d.

    Each dict is *exactly* the keyword arguments passed to find_max_facts (via
    find_max_facts_modal), so the fan-out is a single .map() over the list.
    """
    configs = []
    for d in d_values:
        name = f"{experiment_dir}_{model_type}_d{d}"

        s = copy.deepcopy(base_settings)
        s.input_vocab_size = 2 * d
        s.output_vocab_size = d
        s.d_residual = d
        s.d_ff = d

        # Precision grows ~4x per size step (capacity grows about that much).
        precision = 8 if d == 16 else 8 * 4 if d == 32 else 8 * 4 * 4 if d == 64 else 8 * 4 * 4 * 4

        configs.append(dict(
            settings=s,
            log_to_wandb=log_to_wandb,
            wandb_log_every=10,
            wandb_group=name,
            patience=patience,
            n_epochs=n_epochs,
            lr=lr,
            precision=precision,
            verbose=True,
            # name=name binds the current d's name; a bare lambda would late-bind
            # and give every config the last d's name.
            name_function=lambda settings, name=name: f"{name}_{settings.n_facts}",
            target_accuracy=target_accuracy,
            number_of_attempts=number_of_attempts,
            threshold_to_continue=threshold_to_continue,
            loss_type=loss_type,
            device=device,
        ))
    return configs


# ── Driver: runs locally, orchestrates the Modal fan-out ─────────────────────
@app.local_entrypoint()
def main():
    os.makedirs(experiment_dir, exist_ok=True)
    configs = build_configs()

    log_path = build_log_path()
    # reserve the log file name; create it only if it doesn't already exist (never truncate)
    if not os.path.exists(f"{log_path}.jsonl"):
        with open(f"{log_path}.jsonl", "x"):
            pass
    print(f"Log path: {log_path}.jsonl")

    print(f"Launching {len(configs)} capacity searches on Modal "
          f"(one per d={d_values}); attempts run in parallel on GPU inside each.\n")

    # OUTER fan-out across d. ONE positional iterator: a list of config dicts, each
    # = the kwargs for find_max_facts. find_max_facts_modal adds use_modal=True, so
    # each search nests its attempts onto GPUs.
    max_facts_list = list(find_max_facts_modal.map(configs))

    # Log results locally (back on your laptop).
    for config, max_facts in zip(configs, max_facts_list):
        name = config["wandb_group"]
        settings = config["settings"]
        precision = config["precision"]
        print(f"  {name}: max_facts = {max_facts}")
        log_result(name, max_facts, settings, log_path, extra={
            "wandb_group": name,
            "n_epochs": n_epochs,
            "lr": lr,
            "patience": patience,
            "target_accuracy": target_accuracy,
            "threshold_to_continue": threshold_to_continue,
            "precision": precision,
            "number_of_attempts": number_of_attempts,
            "loss_type": loss_type,
        })

    print(f"\nDone. Results written under {experiment_dir}/.")


# ── Enabling W&B later (optional) ────────────────────────────────────────────
# Training runs in containers, so wandb.init needs your API key there:
#   1. Create the secret once:   modal secret create wandb WANDB_API_KEY=<your-key>
#   2. In capacity_search.py, add to the _try_n_facts_modal decorator:
#          secrets=[modal.Secret.from_name("wandb")]
#   3. Set log_to_wandb = True above.