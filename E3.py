
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

testing = False
model_type = 'full' # 'full' or 'reduced'

log_to_wandb = True
lr = [1e-2, 3e-3, 1e-3]
patience = 5000
n_epochs = 50000
target_accuracy = 'accuracy'
threshold_for_continued_search = 0.99
number_of_attempts = 3
log_path = os.path.join(experiment_dir, f"{model_type}_model_log")


if testing:
    log_to_wandb = False
    lr = [1e-2]
    patience = 10
    n_epochs = 100
    precision = 20
    log_path = os.path.join(experiment_dir, "test_log")


if model_type == 'reduced':
    settings = ModelSettings(attention=False, 
                             ff=True, 
                             bias=True, 
                             norms=False, 
                             ff_residual=False,
                             ff_activation_type='ReLU')
    
elif model_type == 'full':
    settings = ModelSettings(attention=True, 
                             ff=True, 
                             bias=True, 
                             norms=True, 
                             ff_residual=True,
                             ff_activation_type='GELU')
else:
    raise ValueError("Invalid model type. Choose 'full' or 'reduced'.")



def run_sub_experiment(settings: ModelSettings, name: str):

    print(f"Running capacity search for: {name}")

    def name_function(settings: ModelSettings) -> str:
        return f"{name}_{settings.n_facts}"
                
    max_facts = find_max_facts(settings, log_to_wandb=log_to_wandb, wandb_log_every=10, wandb_group=name,
                                patience=patience, n_epochs=n_epochs, lr=lr, precision=precision,
                                verbose = True, name_function=name_function, target_accuracy=target_accuracy,
                                number_of_attempts=number_of_attempts, 
                                threshold_for_continued_search=threshold_for_continued_search)
    
    print(f"Max facts learned: {max_facts}\n")
    
    log_result(name, max_facts, settings, log_path, extra={
        "wandb_group": name,
        "n_epochs": n_epochs,
        "lr": lr,
        "patience": patience,
        "target_accuracy": target_accuracy,
        "threshold_for_continued_search": threshold_for_continued_search,
    })



for d in [16, 32, 64, 128]:
    settings.input_vocab_size = 2*d
    settings.output_vocab_size = d
    settings.d_residual = d
    settings.d_ff = d
    precision = d//2

    name = f"{experiment_dir}_{model_type}_d{d}"
    run_sub_experiment(settings, name)








# %%
