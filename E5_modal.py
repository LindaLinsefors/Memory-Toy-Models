"""E5 architecture-sweep experiment, running on Modal (GPU).

Run it with:
    python -m modal run E5_modal.py

E5 searches the max learnable facts for every architecture variant (attention/qk
x bias x norms x ff_residual x activation, plus an ff=False block) at a fixed
model size. The Modal version fans those variants out:

    main()  (your laptop)
      └── find_max_facts_modal.map(over variants)   ← one CPU container per variant
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
experiment_dir = "E5"
testing = True

loss_type = "CE"            # 'BCE' or 'CE'
any_all_most = "most"       # 'any', 'all', or 'most' (only used on the Modal path)

log_to_wandb = False        # must stay False on Modal without a Secret (see bottom)
lr = [1e-2, 3e-3, 1e-3]
patience = 5000
n_epochs = 50000
precision = 8
target_accuracy = "accuracy"
threshold_to_continue = 0.99
number_of_attempts = 11
verbose = False

if loss_type == "CE":
    target_accuracy = "best_guess_accuracy"
    threshold_to_continue = 0.95

if testing:
    lr = [1e-2]
    patience = 10
    n_epochs = 100
    precision = 20


def build_log_path() -> str:
    if testing:
        return os.path.join(experiment_dir, "test_log")
    log_path = os.path.join(experiment_dir, "experiment_log")
    log_path += f"_{loss_type}_{any_all_most}"
    # Make the log path unique to avoid mixing with previous runs (mirrors E5).
    i = 1
    while os.path.exists(f"{log_path}_({i}).jsonl"):
        i += 1
    return f"{log_path}_({i})"


def build_configs():
    """Replicate E5's architecture sweep as a list of config dicts.

    Each dict is *exactly* the keyword arguments passed to find_max_facts (via
    find_max_facts_modal). We mirror E5's single shared-and-mutated `settings`
    object and deep-copy it at each "run" site, so every config captures exactly
    the variant E5 would have run (including state carried between the two blocks).
    """
    suffix = "_CE" if loss_type == "CE" else ""

    settings = ModelSettings(
        seq_len=2,
        input_vocab_size=32,
        output_vocab_size=16,
        d_residual=16,
        n_heads=1,
        d_ff=16,
    )

    configs = []

    def add_config(name: str):
        s = copy.deepcopy(settings)
        configs.append(dict(
            settings=s,
            log_to_wandb=log_to_wandb,
            wandb_log_every=10,
            wandb_group=name,
            patience=patience,
            n_epochs=n_epochs,
            lr=lr,
            precision=precision,
            verbose=verbose,
            # name=name binds the current variant's name (avoids late-binding).
            name_function=lambda st, name=name: f"{name}_{st.n_facts}{suffix}",
            target_accuracy=target_accuracy,
            number_of_attempts=number_of_attempts,
            threshold_to_continue=threshold_to_continue,
            loss_type=loss_type,
            any_all_most=any_all_most,
        ))

    # Block 1: ff = True
    settings.ff = True
    for attention, qk in [(False, False), (True, False), (True, True)]:
        settings.attention = attention
        settings.qk_is_one = not qk

        for bias in [False, True]:
            settings.bias = bias

            for norms in [False, True]:
                settings.norms = norms

                for ff_residual in [False, True]:
                    settings.ff_residual = ff_residual

                    for ff_activation_type in ["GELU", "ReLU"]:
                        settings.ff_activation_type = ff_activation_type

                        name = f"{experiment_dir}_{'attn' if attention else ''}_{'qk' if qk else ''}_ff_{'norms' if norms else ''}_{'bias' if bias else ''}_{'ffres' if ff_residual else ''}_{ff_activation_type}"
                        add_config(name)

    # Block 2: ff = False
    settings.ff = False
    for attention, qk in [(False, False), (True, False), (True, True)]:
        settings.attention = attention
        settings.qk_is_one = not qk

        for norms in [False, True]:
            settings.norms = norms

            name = f"{experiment_dir}_{'attn' if attention else ''}_{'qk' if qk else ''}__{'norms' if norms else ''}"
            add_config(name)

    return configs


# ── Driver: runs locally, orchestrates the Modal fan-out ─────────────────────
@app.local_entrypoint()
def main():
    os.makedirs(experiment_dir, exist_ok=True)
    log_path = build_log_path()
    configs = build_configs()

    print(f"Launching {len(configs)} capacity searches on Modal "
          f"(one per architecture variant); attempts run in parallel on GPU inside each.\n")

    # OUTER fan-out across variants. ONE positional iterator: a list of config
    # dicts, each = the kwargs for find_max_facts. find_max_facts_modal adds
    # use_modal=True, so each search nests its attempts onto GPUs.
    max_facts_list = list(find_max_facts_modal.map(configs))

    # Log results locally (back on your laptop).
    for config, max_facts in zip(configs, max_facts_list):
        name = config["wandb_group"]
        settings = config["settings"]
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
