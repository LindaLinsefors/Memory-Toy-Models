
#%%
# Test how maximum number of learnable facts scales with model size,
# for few different architecture variants.

import os
import importlib
import models
import capacity_search
import log
import device

importlib.reload(models)
importlib.reload(capacity_search)
importlib.reload(log)
importlib.reload(device)

from models import *
from capacity_search import *
from log import *

# Set the torch default device explicitly (cuda if available, else cpu).
# Override interactively with e.g.  os.environ["MTM_DEVICE"] = "cpu"  before this line.
active_device = device.setup_default_device()
print(f"Using device: {active_device}")

experiment_dir = "E4"
os.makedirs(experiment_dir, exist_ok=True)

#%%

testing = False
duplicate = True

log_to_wandb = True
lr = [1e-2, 3e-3, 1e-3]
patience = 5000
n_epochs = 50000
target_accuracy = 'accuracy'
threshold_to_continue = 0.99
number_of_attempts = 3
log_path = os.path.join(experiment_dir, "log")

loss_type = 'CE' # 'BCE' or 'CE'

if loss_type == 'CE':
    target_accuracy = 'best_guess_accuracy'
    threshold_to_continue = 0.95
    log_path += '_CE'

if duplicate:
    log_path += '_duplicate'


if testing:
    log_to_wandb = False
    lr = [1e-2]
    patience = 10
    n_epochs = 100
    log_path = os.path.join(experiment_dir, "test_log")



settings = ModelSettings(attention=False, 
                            ff=False,
                            norms=False,)

    



def run_sub_experiment(settings: ModelSettings, name: str, precision: int):

    print(f"Running capacity search for: {name}")

    def name_function(settings: ModelSettings) -> str:
        return f"{name}_{settings.n_facts}"
                
    max_facts = find_max_facts(settings, log_to_wandb=log_to_wandb, wandb_log_every=10, wandb_group=name,
                                patience=patience, n_epochs=n_epochs, lr=lr, precision=precision,
                                verbose = True, name_function=name_function, target_accuracy=target_accuracy,
                                number_of_attempts=number_of_attempts, 
                                threshold_to_continue=threshold_to_continue)
    
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



for d in [16, 32, 64, 128]:
    settings.input_vocab_size = 2*d
    settings.output_vocab_size = d
    settings.d_residual = d
    settings.d_ff = d

    #precision = d//2  #This was used for the run recored in full_model_log.txt, but this is way to high precision for the larger models.
    
    #Increase precision with a factor 4 for each model size, since the max facts increases by about that much.
    precision = 8 if d == 16   else 8*4 if d == 32    else 8*4*4 if d == 64      else 8*4*4*4    

    name = f"{experiment_dir}_d{d}"
    run_sub_experiment(settings, name, precision=precision)








# %%




# %%

#Import all data in folder E4, and make a log-log plot. 
#Exclude files with "test" in the name.

import matplotlib.pyplot as plt

#list all files in folder E4
files = os.listdir(experiment_dir)

for file in files:
    if file.endswith(".jsonl") and "test" not in file:
        log_path = os.path.join(experiment_dir, file[:-6])  # strip .jsonl
        data = load_results(log_path)
        if not data:
            continue
        #Make log-log plot of max_facts vs d_residual
        d_residual = [d['settings']['d_residual'] for d in data]
        max_facts = [d['max_facts'] for d in data]
        plt.loglog(d_residual, max_facts, '-o', label=file)

ax = plt.gca()
ax.set_xticks([16, 32, 64, 128])
ax.set_xticks([], minor=True)
ax.set_xticklabels([16, 32, 64, 128])
ax.set_yticks([64, 128, 256, 512, 1024, 2048])
ax.set_yticks([], minor=True)
ax.set_yticklabels([64, 128, 256, 512, 1024, 2048])
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.xlabel("d_residual")
plt.ylabel("max_facts")
plt.legend()
plt.show()
# %%
