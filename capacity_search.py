import copy


import torch
import torch.nn.functional as F

import wandb

from models import ModelSettings, MemoryToyModel, train_model

import modal
app = modal.App("capacity-search")

# The image the Modal containers run in. The default image has none of our deps,
# so we install torch + wandb and ship our local modules so `from models import ...`
# works inside the container. (CPU torch here; for GPU add a CUDA build + gpu=... .)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "wandb", "numpy")
    .add_local_python_source("models", "device", "log")
)


def name_function_n_facts(settings: ModelSettings) -> str:
    """Generate a descriptive name for a run based on the number of facts."""
    return f"n_facts={settings.n_facts}"


def evaluate_model(model, target_accuracy='accuracy', loss_type: str = 'BCE') -> bool:
    """Check whether a trained model has learned all its facts perfectly."""
    model.eval()
    inputs = model.facts["inputs"]
    targets = model.facts["targets"]

    if loss_type not in {'BCE', 'CE'}:
        raise ValueError("loss_type must be either 'BCE' or 'CE'.")

    with torch.no_grad():
        logits = model(inputs)
        best_guess_accuracy = (logits.argmax(dim=-1) == targets).float().mean().item()
        if loss_type == 'BCE':
            one_hot = F.one_hot(targets, model.settings.output_vocab_size)
            accuracy = (one_hot.bool() == (logits > 0)).float().mean().item()
        else:
            accuracy = best_guess_accuracy

    if target_accuracy == 'accuracy':
        return accuracy == 1.0
    elif target_accuracy == 'best_guess_accuracy':
        return best_guess_accuracy == 1.0


def find_max_facts(settings: ModelSettings,
                   precision: int = 1,
                   n_epochs: int = 2000,
                   lr: list[float] = [1e-2],
                   optimizer_type: str = 'Adam',
                   grad_clip_norm=None,
                   patience: int = 100,
                   number_of_attempts: int = 1,
                   log_to_wandb: bool = True, wandb_log_every: int = 10,
                   wandb_group: str = 'capacity_search',
                   verbose: bool = True,
                   name_function = name_function_n_facts,
                   target_accuracy: str = 'accuracy',
                   threshold_to_continue: float = 0.99,
                   loss_type: str = 'BCE',
                   use_modal: bool = False,
                   any_all_most: str = 'any') -> int:
    """Binary search for the maximum number of facts a model architecture can learn.

    Args:
        settings:      Base ModelSettings (n_facts will be overridden per candidate).
        precision:     Stop searching when hi - lo < precision.
        n_epochs:      Max training epochs per run.
        patience:      Early-stopping patience (epochs without accuracy improvement).
        lr:            List of learning rates to try.
        number_of_attempts: Number of independent attempts per candidate n_facts.
                       A single success is sufficient; only all-fail counts as failure.
        log_to_wandb:  Whether to log individual runs to W&B.
        wandb_group:   W&B group name for the search.
        wandb_log_every: How often to log to W&B (in epochs).
        verbose:       Print progress during the search.
        target_accuracy:     Metric to train for ('accuracy' or 'best_guess_accuracy').
        threshold_to_continue: 
                       Minimum accuracy required to continue searching with a lower learning rate.
        loss_type:     Either 'BCE' or 'CE' for the training loss used in each trial.
        use_modal:     If True, run the per-candidate attempts in parallel on Modal
                       (via _try_n_facts_modal.map). If False, run them locally in a loop.
        any_all_most:  Only used if use_modal is True. 
                       If 'any', a single successful attempt counts as success. If 'all',
                       all attempts must succeed. If 'most', a majority of attempts must succeed.
    Returns:
        The maximum n_facts for which the model achieved perfect accuracy
        (within the given precision).
    """
    max_possible = settings.input_vocab_size ** settings.seq_len
    lo, hi = 1, max_possible
    best = 0  # highest n_facts confirmed learnable

    if verbose:
        print(f"Searching for max learnable facts in [{lo}, {hi}]")
        print()

    while hi - lo >= precision:
        mid = (lo + hi) // 2

        if verbose:
            print(f"Trying n_facts = {mid} ...")

        if not use_modal:
            learned = False
            for attempt in range(1, number_of_attempts + 1):
                if verbose and number_of_attempts > 1:
                    print(f"  Attempt {attempt}/{number_of_attempts}")
                success = _try_n_facts(settings, mid,
                                    n_epochs=n_epochs, lr=lr, 
                                    optimizer_type=optimizer_type, grad_clip_norm=grad_clip_norm, 
                                    patience=patience,
                                    log_to_wandb=log_to_wandb,
                                    wandb_group=wandb_group,
                                    wandb_log_every=wandb_log_every,
                                    verbose=verbose,
                                    name_function=name_function,
                                    target_accuracy=target_accuracy,
                                    threshold_to_continue=threshold_to_continue,
                                    loss_type=loss_type)
                if success:
                    learned = True
                    break

        else:
            # Run all `number_of_attempts` attempts in PARALLEL on Modal.
            # .map() wants one positional iterator per positional arg of the
            # function. The only positional arg of _try_n_facts is base_settings,
            # so we repeat it `number_of_attempts` times (that sets the count).
            # Everything else is identical across attempts -> pass via kwargs,
            # which .map() broadcasts to every call.
            successes = list(_try_n_facts_modal.map(
                [settings] * number_of_attempts,   # -> base_settings, N copies
                kwargs=dict(
                    n_facts=mid,
                    n_epochs=n_epochs,
                    lr=lr,
                    optimizer_type=optimizer_type,
                    grad_clip_norm=grad_clip_norm,
                    patience=patience,
                    log_to_wandb=log_to_wandb,
                    wandb_group=wandb_group,
                    wandb_log_every=wandb_log_every,
                    verbose=verbose,
                    name_function=name_function,
                    target_accuracy=target_accuracy,
                    threshold_to_continue=threshold_to_continue,
                    loss_type=loss_type,
                ),
            ))
            if any_all_most == 'any':
                learned = any(successes)   # "any attempt wins"
            elif any_all_most == 'all':
                learned = all(successes)   # "all attempts must succeed"
            elif any_all_most == 'most':
                learned = sum(successes) > len(successes) // 2   # "majority wins"

        if learned:
            best = mid
            lo = mid + 1
            if verbose:
                print(f"✓  learned all {mid} facts. Now searching: {lo} - {hi}")
        else:
            hi = mid - 1
            if verbose:
                print(f"✗  failed to learn {mid} facts. Now searching: {lo} - {hi}")
            
    if verbose:
        print(f"\nMax learnable facts: {best}")

    return best

@app.function(image=image, timeout=86400)  # 24h cap: outer waits on the whole per-d search
def find_max_facts_modal(config: dict) -> int:
    """Run one capacity search on Modal.

    `config` is exactly the keyword arguments for find_max_facts (settings,
    precision, lr, wandb_group, name_function, ...). We only add use_modal=True,
    which makes each search nest its attempts onto GPU containers.
    """
    return find_max_facts(use_modal=True, **config)



def _try_n_facts(base_settings: ModelSettings,
                 n_facts: int,
                 n_epochs: int,
                 lr: list[float],
                 optimizer_type: str,
                 grad_clip_norm: int|None,
                 patience: int,
                 log_to_wandb: bool,
                 wandb_group: str,
                 wandb_log_every: int,
                 verbose: bool,
                 name_function: callable,
                 target_accuracy: str,
                 threshold_to_continue: float,
                 loss_type: str ) -> bool:
    """Train a model with the given n_facts. Returns True if it achieves perfect accuracy."""
    trial_settings = copy.deepcopy(base_settings)
    trial_settings.n_facts = n_facts

    model = MemoryToyModel(trial_settings)

    if log_to_wandb:
        wandb.init(
            project="Memory Toy Models",
            group=wandb_group,
            config=vars(trial_settings),
            reinit=True,
            name=name_function(trial_settings)
        )

    for lr_value in lr:
        success, best_accuracy = train_model(
                                    model, n_epochs=n_epochs, lr=lr_value, 
                                    optimizer_type=optimizer_type, grad_clip_norm=grad_clip_norm,
                                    log_to_wandb=log_to_wandb, wandb_log_every=wandb_log_every, 
                                    wandb_finish=False, wandb_group=wandb_group,
                                    early_stopping=True, patience=patience, verbose=verbose,
                                    target_accuracy=target_accuracy, loss_type=loss_type)
        if success:
            break  # stop if we found a learning rate that works
        if best_accuracy < threshold_to_continue:
            break  # stop if accuracy is very low, unlikely to improve with more training

    if log_to_wandb:
        wandb.log({"capacity_search/n_facts": n_facts,
                    "capacity_search/success": success,
                    "capacity_search/best_accuracy": best_accuracy})
        wandb.finish()

    return success

@app.function(image=image, gpu="T4", timeout=10800)  # 1h cap: one full 50k-epoch attempt
def _try_n_facts_modal(*args, **kwargs) -> bool:
    """Variant of _try_n_facts that runs on Modal, on a GPU.

    This is where the actual training happens, so it gets the GPU. models.py
    sets the default device to cuda-if-available, so on this container the model
    trains on the GPU automatically. Change gpu="T4" to e.g. "A10G"/"A100" if you
    want a bigger card (overkill for these tiny models, but easy to switch).
    """
    return _try_n_facts(*args, **kwargs)



if __name__ == "__main__":
    # Example: search capacity for the default architecture
    settings = ModelSettings()
    max_facts = find_max_facts(settings, log_to_wandb=False)
    print(f"Result: {max_facts}")
