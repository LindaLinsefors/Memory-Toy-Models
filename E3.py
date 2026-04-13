
#%%
# Test how maximum number of learnable facts scales with model size, 
# for few different architecture variants.

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

experiment_dir = "E3"
os.makedirs(experiment_dir, exist_ok=True)

testing = True

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


target_accuracy = 'accuracy'
log_path = os.path.join(experiment_dir, "experiment_log")


settings = ModelSettings(
    seq_len = 2,
    n_heads = 1,

    ff_activation_type = 'GELU',
    norms = True,
    bias = True,
    ff_residual = True)



def run_sub_experiment(settings: ModelSettings, name: str):
    print(f"Running capacity search for: {name}")

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


for d in [16, 32, 64, 128]:
    
    settings.input_vocab_size = 2*d,
    settings.output_vocab_size = d,
    settings.d_residual = d,
    settings.d_ff = d

    if not testing:
        precision = (d//8) ** 2

    for attention in [False, True]:
        settings.attention = attention
        for ff in [False, True]:
            settings.ff = ff

            name = f"{experiment_dir}_{d}D_{'attn' if attention else ''}_{'ff' if ff else ''}"
            run_sub_experiment(settings, name)



# %%
