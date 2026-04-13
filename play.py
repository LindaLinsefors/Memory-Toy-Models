

#%%
#Re-Inmport function for Playground
import importlib
import models
import capacity_search
importlib.reload(models)
importlib.reload(capacity_search)
from models import *
from capacity_search import *
import time

# %%

settings = ModelSettings()
max_facts = find_max_facts(settings, log_to_wandb=True, wandb_group="playground")

# %%


max_facts = find_max_facts(settings, log_to_wandb=True, wandb_group="playground",
                           patience=1000, n_epochs=10000)
# %%
max_facts = find_max_facts(settings, log_to_wandb=True, wandb_group="playground_patience_300",
                           patience=300, n_epochs=10000)
# %%
group = "playground_2"

time_start = time.time()
max_facts = find_max_facts(settings, log_to_wandb=True, wandb_group=group,
                           patience=300, n_epochs=10000, lr=[1e-2, 1e-3])
time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")
# %%
group = "playground_3"

time_start = time.time()
max_facts = find_max_facts(settings, log_to_wandb=True, wandb_group=group,
                           patience=1000, n_epochs=10000, lr=[1e-2, 1e-3])
time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")
# %%
group = "playground_4"

time_start = time.time()
max_facts = find_max_facts(settings, log_to_wandb=True, wandb_log_every=10, wandb_group=group,
                           patience=10000, n_epochs=100000, lr=[1e-2, 3e-3, 1e-3]) 
                           
time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")
# %%


group = "playground_5"

time_start = time.time()
max_facts = find_max_facts(settings, log_to_wandb=True, wandb_log_every=1, wandb_group=group,
                           patience=10000, n_epochs=100000, lr=[1e-2, 3e-3, 1e-3])
time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")
# %%


group = "playground_6"

time_start = time.time()
max_facts = find_max_facts(settings, log_to_wandb=True, wandb_log_every=1, wandb_group=group,
                           patience=1000, n_epochs=10000, lr=[1e-2, 3e-3, 1e-3])
time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")

# %%


group = "playground_7"

time_start = time.time()
max_facts = find_max_facts(settings, log_to_wandb=True, wandb_log_every=10, wandb_group=group,
                           patience=1000, n_epochs=10000, lr=[1e-2, 3e-3, 1e-3])
time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")
# %%


group = "playground_8"

time_start = time.time()
max_facts = find_max_facts(settings, log_to_wandb=False, wandb_log_every=10, wandb_group=group,
                           patience=5000, n_epochs=50000, lr=[1e-2, 3e-3, 1e-3])
time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")
# %%


group = "compare_with_old_experments_extra_patience"

settings_old = ModelSettings(seq_len=2, 
                             input_vocab_size=32, 
                             output_vocab_size=16, 
                             d_residual=16, 
                             d_ff=16,
                             attention=False, 
                             ff=True, bias=True, 
                             norms = False, 
                             ff_residual=False, 
                             ff_activation_type='GELU')

time_start = time.time()
max_facts = find_max_facts(settings_old, log_to_wandb=True, wandb_log_every=10, wandb_group=group,
                           patience=30000, n_epochs=100000, lr=[1e-2, 3e-3, 1e-3])
time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")
# %%
group = "compare_with_old_experments_best_guess_accuracy"

settings_old = ModelSettings(seq_len=2, 
                             input_vocab_size=32, 
                             output_vocab_size=16, 
                             d_residual=16, 
                             d_ff=16,
                             attention=False, 
                             ff=True, bias=True, 
                             norms = False, 
                             ff_residual=False, 
                             ff_activation_type='GELU')

time_start = time.time()
max_facts = find_max_facts(settings_old, log_to_wandb=True, wandb_log_every=10, wandb_group=group,
                           patience=5000, n_epochs=50000, lr=[1e-2, 3e-3, 1e-3], target_accuracy='best_guess_accuracy')
time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")
# %%
group = "compare_with_old_experments_best_guess_accuracy_fixed_code"

settings_old = ModelSettings(seq_len=2, 
                             input_vocab_size=32, 
                             output_vocab_size=16, 
                             d_residual=16, 
                             d_ff=16,
                             attention=False, 
                             ff=True, bias=True, 
                             norms = False, 
                             ff_residual=False, 
                             ff_activation_type='GELU')

time_start = time.time()
max_facts = find_max_facts(settings_old, log_to_wandb=True, wandb_log_every=10, wandb_group=group,
                           patience=5000, n_epochs=50000, lr=[1e-2, 3e-3, 1e-3], target_accuracy='best_guess_accuracy')
time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")
# %%
