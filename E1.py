
#%%
#Experiment 1: Search for maximum capacity of every architecture variant.
#All runs with [attention = False] are wrong due to an old bug. 

import torch
torch.set_default_device("cuda")

import os
import importlib
import models
import capacity_search
import log

importlib.reload(models)
importlib.reload(capacity_search)
importlib.reload(log)

from models import *
from capacity_search import *
from log import *

experiment_dir = "E1"
os.makedirs(experiment_dir, exist_ok=True)



testing = False 
extra_patience = False 
best_guess_accuracy = True


if testing:
    log_to_wandb = False
    lr = [1e-2]
    patience = 10
    n_epochs = 100
    precision = 20
    log_path = os.path.join(experiment_dir, "test_log")
else:
    log_to_wandb = True
    lr = [1e-2, 3e-3, 1e-3]
    patience = 5000
    n_epochs = 50000
    precision = 8
    target_accuracy = 'accuracy'
    log_path = os.path.join(experiment_dir, "experiment_log")

    if extra_patience:
        patience = 30000
        n_epochs = 100000
        log_path = os.path.join(experiment_dir, "experiment_log_extra_patience")

    if best_guess_accuracy:
        log_path += "_best_guess_accuracy"
        target_accuracy = 'best_guess_accuracy'

settings = ModelSettings(
    seq_len = 2,
    input_vocab_size = 32,
    output_vocab_size = 16,
    d_residual = 16,
    n_heads = 1,
    d_ff = 16
)

def run_sub_experiment(settings: ModelSettings, name: str):

    print(f"Running capacity search for: {name}")


    if best_guess_accuracy:
        def name_function(n_facts: int) -> str:
            return f"{name}_{n_facts}_bgacc"
    else:
        def name_function(n_facts: int) -> str:
            return f"{name}_{n_facts}"
                

    max_facts = find_max_facts(settings, log_to_wandb=log_to_wandb, wandb_log_every=10, wandb_group=name,
                                patience=patience, n_epochs=n_epochs, lr=lr, precision=precision,
                                verbose = False, name_function=name_function, target_accuracy=target_accuracy)
    
    print(f"Max facts learned: {max_facts}\n")
    
    log_result(name, max_facts, settings, log_path, extra={
        "wandb_group": name,
        "n_epochs": n_epochs,
        "lr": lr,
        "patience": patience,
        "target_accuracy": target_accuracy,
    })


ff = True
settings.ff = ff

for attention in [False, True]:
    settings.attention = attention

    for bias in [False, True]:
        settings.bias = bias

        for norms in [False, True]:
            settings.norms = norms

            for ff_residual in [False, True]:
                settings.ff_residual = ff_residual

                for ff_activation_type in ['GELU', 'ReLU']:
                    settings.ff_activation_type = ff_activation_type

                    name = f"{experiment_dir}_{'attn' if attention else ''}_ff_{'norms' if norms else ''}_{'bias' if bias else ''}_{'ffres' if ff_residual else ''}_{ff_activation_type}"
                    run_sub_experiment(settings, name)


ff = False
settings.ff = ff

for attention in [False, True]:
    settings.attention = attention

    for norms in [False, True]:
        settings.norms = norms
        
        name = f"{experiment_dir}_{'attn' if attention else ''}__{'norms' if norms else ''}"
        run_sub_experiment(settings, name)

# %%
