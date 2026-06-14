
#%%
#Experiment 5: Search for maximum capacity of every architecture variant. 
#Updated version of E2

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

experiment_dir = "E5"
os.makedirs(experiment_dir, exist_ok=True)



testing = True


loss_type = 'CE'            # 'BCE' or 'CE'
any_all_most = 'most'       # 'any', 'all', or 'most'


log_to_wandb = False
lr = [1e-2, 3e-3, 1e-3]
patience = 5000
n_epochs = 50000
precision = 8
target_accuracy = 'accuracy'
threshold_to_continue = 0.99
number_of_attempts = 11
verbose = False

log_path = os.path.join(experiment_dir, "experiment_log")
log_path += f"_{loss_type}_{any_all_most}"

#Make sure the log path is unique to avoid mixing with previous runs
i = 1
while os.path.exists(f"{log_path}_({i})"):
    i += 1
log_path = f"{log_path}_({i})"

if loss_type == 'CE':
    target_accuracy = 'best_guess_accuracy'
    threshold_to_continue = 0.95


if testing:
    log_to_wandb = False
    lr = [1e-2]
    patience = 10
    n_epochs = 100
    precision = 20
    log_path = os.path.join(experiment_dir, "test_log")

 

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


    if loss_type == 'CE':
        def name_function(settings: ModelSettings) -> str:
            return f"{name}_{settings.n_facts}_CE"
    else:
        def name_function(settings: ModelSettings) -> str:
            return f"{name}_{settings.n_facts}"
                

    max_facts = find_max_facts(settings, log_to_wandb=log_to_wandb, wandb_log_every=10, wandb_group=name,
                                patience=patience, n_epochs=n_epochs, lr=lr, precision=precision,
                                verbose = verbose, name_function=name_function, target_accuracy=target_accuracy,
                                number_of_attempts=number_of_attempts, 
                                threshold_to_continue=threshold_to_continue, loss_type=loss_type)
    
    print(f"Max facts learned: {max_facts}\n")
    
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


ff = True
settings.ff = ff

for attention, qk in [(False, False), (True, False), (True, True)]:
    settings.attention = attention
    settings.qk_is_one = not qk

    for bias in [False, True]:
        settings.bias = bias

        for norms in [False, True]:
            settings.norms = norms

            for ff_residual in [False, True]:
                settings.ff_residual = ff_residual

                for ff_activation_type in ['GELU', 'ReLU']:
                    settings.ff_activation_type = ff_activation_type

                    name = f"{experiment_dir}_{'attn' if attention else ''}_{'qk' if qk else ''}_ff_{'norms' if norms else ''}_{'bias' if bias else ''}_{'ffres' if ff_residual else ''}_{ff_activation_type}"
                    run_sub_experiment(settings, name)


ff = False
settings.ff = ff

for attention, qk in [(False, False), (True, False), (True, True)]:
    settings.attention = attention
    settings.qk_is_one = not qk

    for norms in [False, True]:
        settings.norms = norms
        
        name = f"{experiment_dir}_{'attn' if attention else ''}_{'qk' if qk else ''}__{'norms' if norms else ''}"
        run_sub_experiment(settings, name)



# %%
5
# %%
