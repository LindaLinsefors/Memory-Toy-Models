

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

torch.set_default_device("cuda")
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
5
# %%

group = "test clip, AdamW, smoot, lr (long run)"

wandb.finish()
settings = ModelSettings(seq_len=2, 
                             input_vocab_size=64, 
                             output_vocab_size=32, 
                             d_residual=32, 
                             d_ff=32,
                             attention=True, 
                             ff=True, bias=True, 
                             norms = False, 
                             ff_residual=False, 
                             ff_activation_type='GELU',
                             n_facts=2320)

for _ in range(3):  # repeat the whole grid search 3 times to check for consistency
    for smoothing in [0.05, 0.01, None]:
        for grad_clip_norm in [1.0, None]:
            for optimizer_type in ['AdamW', 'Adam']:
                for lr in [1e-2, 1e-3]:
                    name = f"clip_{grad_clip_norm}_{optimizer_type}_smoot_{smoothing}_lr_{lr}"
                    print(f'Testing grad_clip_norm={grad_clip_norm}, optimizer_type={optimizer_type}, smoothing={smoothing}, lr={lr}')
                    print(f'wandb name = {name}')

                    time_start = time.time()
                    model = MemoryToyModel(settings)
                    success, best_accuracy = train_model(
                            model, n_epochs=20000, lr=lr, 
                            optimizer_type=optimizer_type, grad_clip_norm=grad_clip_norm, smoothing=smoothing,
                            log_to_wandb=True, wandb_log_every=1, wandb_group=group, wandb_name=name)

                    time_end = time.time()
                    print(f"Time taken: {time_end - time_start} seconds\n")

#%%
group = "testing lr"

wandb.finish()
settings = ModelSettings(seq_len=2, 
                             input_vocab_size=64, 
                             output_vocab_size=32, 
                             d_residual=32, 
                             d_ff=32,
                             attention=True, 
                             ff=True, bias=True, 
                             norms = False, 
                             ff_residual=False, 
                             ff_activation_type='GELU',
                             n_facts=2320)

for lr in [1e-1, 1e-2, 1e-3]:
        name = f"lr_{lr}"
        print(f'Testing learning rate={lr}')

        time_start = time.time()
        model = MemoryToyModel(settings)
        success, best_accuracy = train_model(
                model, n_epochs=5000, lr=lr, 
                log_to_wandb=True, wandb_log_every=1, wandb_group=group, wandb_name=name)

        time_end = time.time()
        print(f"Time taken: {time_end - time_start} seconds\n")

# %%


group = "test clip, AdamW, smoot, lr (n_facts = 1000)"

wandb.finish()
settings = ModelSettings(seq_len=2, 
                             input_vocab_size=64, 
                             output_vocab_size=32, 
                             d_residual=32, 
                             d_ff=32,
                             attention=True, 
                             ff=True, bias=True, 
                             norms = False, 
                             ff_residual=False, 
                             ff_activation_type='GELU',
                             n_facts=1000)

for _ in range(3):  # repeat the whole grid search 3 times to check for consistency
    for smoothing in [0.05, 0.01, None]:
        for grad_clip_norm in [1.0, None]:
            for optimizer_type in ['AdamW', 'Adam']:
                for lr in [1e-2, 1e-3]:
                    name = f"clip_{grad_clip_norm}_{optimizer_type}_smoot_{smoothing}_lr_{lr}"
                    print(f'Testing grad_clip_norm={grad_clip_norm}, optimizer_type={optimizer_type}, smoothing={smoothing}, lr={lr}')
                    print(f'wandb name = {name}')

                    time_start = time.time()
                    model = MemoryToyModel(settings)
                    success, best_accuracy = train_model(
                            model, n_epochs=5000, lr=lr, 
                            optimizer_type=optimizer_type, grad_clip_norm=grad_clip_norm, smoothing=smoothing,
                            log_to_wandb=True, wandb_log_every=1, wandb_group=group, wandb_name=name)

                    time_end = time.time()
                    print(f"Time taken: {time_end - time_start} seconds\n")

#%%