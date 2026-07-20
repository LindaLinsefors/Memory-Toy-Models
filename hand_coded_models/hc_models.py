# %%
import torch
import torch.nn.functional as F
torch.set_default_device("cuda")

import matplotlib.pyplot as plt

from models import generate_facts
from hand_coded_models.neuron_assigments import generate_neuron_assignments



class HandCodedModelSettings:
    def __init__(self, input_vocab_size=32, output_vocab_size=16, n_facts=16, seed=42,
                 d_ff=16, n_neurons_per_label=3, 
                 use_top_n_or_top_fraction='top_n', top_n=0, top_fraction=0.2):
        
        # Data dimensions
        self.seq_len = 2  # Fixed for this model
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.n_facts = n_facts
        self.seed = seed  # For reproducibility

        # Internal model settings
        self.d_ff = d_ff
        self.n_neurons_per_label = n_neurons_per_label
        self.use_top_n_or_top_fraction = use_top_n_or_top_fraction  # 'top_n' or 'top_fraction'
        self.top_n = top_n
        self.top_fraction = top_fraction
        self.adjustments = True

def generate_settings(d):
    return HandCodedModelSettings(input_vocab_size = 2*d, 
                                  output_vocab_size = d, 
                                  n_facts = d,
                                  d_ff = d)


def search_best_top_n(d, n_facts, retries=2, metric='accuracy', adjustments=True):
    """Search over values of top_n to find the one giving the best accuracy.

    Builds a model with input_vocab_size=2*d, output_vocab_size=d, d_ff=d and
    the given n_facts, then increases top_n from 0 upward. For each top_n the
    model is built a few times (since construction is stochastic) and the best
    accuracy is kept. The search stops once raising top_n has decreased the
    accuracy twice in a row.

    metric selects which score to track: 'accuracy' or 'best_guess_accuracy'.
    adjustments toggles the model's post-hoc bias/intervention adjustments.

    Returns (best_top_n, best_accuracy).
    """
    if metric not in ('accuracy', 'best_guess_accuracy'):
        raise ValueError("metric must be 'accuracy' or 'best_guess_accuracy'")

    settings = generate_settings(d)
    settings.n_facts = n_facts
    settings.use_top_n_or_top_fraction = 'top_n'
    settings.adjustments = adjustments

    best_top_n = 0
    best_accuracy = 0.0
    prev_accuracy = -1.0
    decreases_in_a_row = 0

    top_n = 0
    while True:
        settings.top_n = top_n

        accuracy_for_top_n = 0.0
        for _ in range(retries):
            model = HandCodedModel(settings)
            accuracy, best_guess_accuracy, _, _ = model.evaluate()
            score = accuracy if metric == 'accuracy' else best_guess_accuracy
            accuracy_for_top_n = max(accuracy_for_top_n, score)
            if accuracy_for_top_n == 1.0:
                break

        if accuracy_for_top_n > best_accuracy:
            best_accuracy = accuracy_for_top_n
            best_top_n = top_n

        if best_accuracy == 1.0:
            break

        if accuracy_for_top_n < prev_accuracy:
            decreases_in_a_row += 1
            if decreases_in_a_row >= 2:
                break
        else:
            decreases_in_a_row = 0

        prev_accuracy = accuracy_for_top_n
        top_n += 1

    return best_top_n, best_accuracy


def search_max_facts(d, accuracy_threshold, retries=2, metric='accuracy', adjustments=True):
    """Find the largest n_facts whose best accuracy still meets a threshold.

    For a model of dimension d, uses search_best_top_n to score a given number
    of facts. n_facts is always a multiple of d (k * d), since generate_facts
    only distributes labels evenly in that case. First grows k exponentially to
    bracket the failure point, then binary searches between the last passing and
    first failing multipliers.

    adjustments toggles the model's post-hoc bias/intervention adjustments.

    Returns (max_facts, best_top_n, best_accuracy) for the largest passing
    n_facts found, or (0, None, None) if even n_facts=d fails.
    """
    def passes(k):
        n_facts = k * d
        top_n, accuracy = search_best_top_n(d, n_facts, retries=retries,
                                            metric=metric, adjustments=adjustments)
        return accuracy >= accuracy_threshold, top_n, accuracy

    # Exponentially grow the multiplier k to bracket: lo passes, hi fails.
    lo = 0  # k=0 (n_facts=0) trivially passes (nothing to learn)
    lo_result = (None, None)
    hi = 1

    while True:
        ok, top_n, accuracy = passes(hi)
        if not ok:
            break
        lo = hi
        lo_result = (top_n, accuracy)
        hi *= 2

    # If even the smallest count (k=1, n_facts=d) fails, nothing passes.
    if lo == 0:
        return 0, None, None

    # Binary search for the boundary in (lo, hi): lo passes, hi fails.
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok, top_n, accuracy = passes(mid)
        if ok:
            lo = mid
            lo_result = (top_n, accuracy)
        else:
            hi = mid

    best_top_n, best_accuracy = lo_result
    return lo * d, best_top_n, best_accuracy

class HandCodedModel:
    def __init__(self, settings: HandCodedModelSettings):
        self.settings = settings
        self.facts = generate_facts(n_facts=settings.n_facts, seq_len=2,
                                    input_vocab_size=settings.input_vocab_size, 
                                    output_vocab_size=settings.output_vocab_size, 
                                    seed=settings.seed)

    
        # This code has diffrent names for the same parameters, becasue I've coppied over code 
        # from other places and I don't want to change it right now. I might clean this up later.
        inputs = self.facts['inputs']
        labels = self.facts['targets']
        S = settings.n_neurons_per_label
        n_labels = settings.output_vocab_size
        hidden_dim = settings.d_ff
        n_vocab = settings.input_vocab_size
        l_input = settings.seq_len

        self.neuron_assignments = generate_neuron_assignments(S = S,
                                                        n_labels = n_labels, 
                                                        hidden_dim = hidden_dim).to("cuda")
        my_assignments = self.neuron_assignments
        

        # List labels for each neuron based on the assignments
        labels_for_neuron = []
        for neuron in range(hidden_dim):
            labels_for_this_neuron = []
            for label in range(n_labels):
                if my_assignments[label, neuron] == 1:
                    labels_for_this_neuron.append(label)
            labels_for_neuron.append(labels_for_this_neuron)
        labels_for_neuron = torch.tensor(labels_for_neuron)

        # List inputs for each neuron 
        inputs_for_neuron = []
        mask = (labels_for_neuron[:,:,None] == labels[None,None,:]).any(dim=1)
        for neuron in range(hidden_dim):
            inputs_for_this_neuron = inputs[mask[neuron]]
            inputs_for_neuron.append(inputs_for_this_neuron)
        inputs_for_neuron = torch.stack(inputs_for_neuron)

        mlp_up = torch.ones(hidden_dim, n_vocab*l_input)

        for neuron in range(hidden_dim):
            unique_f, counts_f = torch.unique(inputs_for_neuron[neuron,:, 0], return_counts=True)
            unique_s, counts_s = torch.unique(inputs_for_neuron[neuron,:, 1], return_counts=True)

            perm = torch.randperm(unique_f.shape[0])
            unique_f = unique_f[perm]; counts_f = counts_f[perm]
            perm = torch.randperm(unique_s.shape[0])
            unique_s = unique_s[perm]; counts_s = counts_s[perm]

            if settings.use_top_n_or_top_fraction == 'top_fraction':
                top_first_inputs = unique_f[torch.argsort(counts_f, descending=True)[:int(len(unique_f)*settings.top_fraction)]]
                top_second_inputs = unique_s[torch.argsort(counts_s, descending=True)[:int(len(unique_s)*settings.top_fraction)]]

            if settings.use_top_n_or_top_fraction == 'top_n':
                top_first_inputs = unique_f[torch.argsort(counts_f, descending=True)[:settings.top_n]]
                top_second_inputs = unique_s[torch.argsort(counts_s, descending=True)[:settings.top_n]]

            remaining_inputs = [inp for inp in inputs_for_neuron[neuron] 
                                if inp[0] not in top_first_inputs 
                                and inp[1] not in top_second_inputs]
            
            mlp_up[neuron, top_first_inputs] = -1
            mlp_up[neuron, top_second_inputs + n_vocab] = -1

            if remaining_inputs:
                remaining_inputs = torch.stack(remaining_inputs)
                mlp_up[neuron, remaining_inputs[:, 0]] = 0
                mlp_up[neuron, remaining_inputs[:, 1] + n_vocab] = 0

        self.up_matrix = mlp_up
        self.down_matrix = - 2.0 * my_assignments
        self.down_bias = torch.ones(n_labels)  # Bias to push towards 1, since we are using negative weights

        if settings.adjustments == True:

            logits, hidden = self.forward(inputs)

            self.logg_intervention_neurons = torch.zeros(n_labels, hidden_dim)
            for label in range(n_labels):
                problems = (logits[:, label] > 0) & (labels != label)
                if problems.any():
                    intervention_neurons = ((hidden[problems]) == 0).any(dim=0) & (hidden[labels == label] > 0).any(dim=0)
                    self.logg_intervention_neurons[label, intervention_neurons] = 1
            self.down_matrix += self.logg_intervention_neurons

            logits, hidden = self.forward(inputs)

            for label in range(n_labels):
                min_true_logit = logits[labels == label, label].min()
                max_false_logit = logits[labels != label, label].max()
                self.down_bias[label] -= (max_false_logit + min_true_logit) / 2


    def forward(self, x):
        first = F.one_hot(x[:,0], num_classes=self.settings.input_vocab_size).float()
        second = F.one_hot(x[:,1], num_classes=self.settings.input_vocab_size).float()
        x = torch.cat([first, second], dim=-1)
        hidden = torch.relu(x @ self.up_matrix.T)
        logits = hidden @ self.down_matrix.T + self.down_bias[None, :]
        return logits, hidden
    
    def evaluate(self):
        logits, hidden = self.forward(self.facts['inputs'])
        one_hot_targets = F.one_hot(self.facts['targets'], self.settings.output_vocab_size)
        accuracy = (one_hot_targets.bool() == (logits > 0)).float().mean().item()
        best_guess_accuracy = self.facts['targets'].eq(logits.argmax(dim=1)).float().mean().item()
        return accuracy, best_guess_accuracy, logits, hidden


#%%

settings = HandCodedModelSettings()
settings.n_facts = 32
model = HandCodedModel(settings)
accuracy, best_guess_accuracy, logits, hidden = model.evaluate()

plt.imshow(hidden.cpu(), aspect='auto', cmap='viridis')
plt.colorbar()
plt.title("Hidden Layer Activations")
plt.xlabel("Hidden Neurons")
plt.ylabel("Facts")
plt.show()

plt.imshow(torch.sign(logits).cpu(), aspect='auto', cmap='viridis')
plt.colorbar()
plt.title("Logits Sign")
plt.xlabel("Output Neurons")
plt.ylabel("Facts")
plt.show()

plt.imshow(model.logg_intervention_neurons.cpu(), aspect='auto', cmap='viridis')
plt.colorbar()
plt.title("Intervention Neurons")
plt.xlabel("Hidden Neurons")
plt.ylabel("Output Neurons")
plt.show()


#%%
settings = HandCodedModelSettings()
for n_facts in [16, 16*2, 16*3, 16*4]:
    settings.n_facts = n_facts
    print(f"\nEvaluating hand with n_facts = {settings.n_facts}\n")

    for top_n in [0, 1]:
        print(f"top_n = {top_n}")

        settings.top_n = top_n

        best_accuracy = 0
        for _ in range(10):
            model = HandCodedModel(settings)
            accuracy, best_guess_accuracy, logits, hidden = model.evaluate()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                if best_accuracy == 1.0:
                    break
        print(f"Best Accuracy: {best_accuracy}")

#%%

verbose = True
max_facts_d = {}

# %%

for d_model in [128, 256, 512, 1024]:
    settings = HandCodedModelSettings(input_vocab_size=2*d_model, 
                                      output_vocab_size=d_model, 
                                      d_ff=d_model,
                                      n_facts=0,
                                      use_top_n_or_top_fraction='top_n', top_n=0)
    
    max_facts = 0
    success = True
    if verbose: print(f"\nEvaluating hand coded model with d_model = {d_model}\n")

    while success:
        success = False
        settings.top_n = 0
        settings.n_facts += d_model
        keep_trying = 2
        best_accuracy = 0

        while keep_trying > 0:
            model = HandCodedModel(settings)
            accuracy_first_try,_ , _, _ = model.evaluate()
            if verbose: print(f"  Accuracy {accuracy_first_try:.8f} with {settings.n_facts} facts and top_n = {settings.top_n}.")

            if accuracy_first_try == 1.0:
                accuracy = accuracy_first_try  
            else:
                model = HandCodedModel(settings)
                accuracy_second_try, _, _ = model.evaluate()
                if verbose: print(f"  Accuracy {accuracy_second_try:.8f} with {settings.n_facts} facts and top_n = {settings.top_n}.")

                accuracy = max(accuracy_first_try, accuracy_second_try)

            if accuracy == 1.0: #Success, we're done with this number of facts.
                max_facts = settings.n_facts
                success = True
                keep_trying = 0
                if verbose: print(f"✓ Learned {settings.n_facts} facts with top_n = {settings.top_n}.")

            elif accuracy >= best_accuracy: #Try again with one higher top_n.
                best_accuracy = accuracy
                settings.top_n += 1 
                keep_trying = 2
            
            elif accuracy < best_accuracy: #Give up on this number of facts.
                settings.top_n += 1 
                keep_trying -= 1


    max_facts_d[d_model] = max_facts
    if verbose: print(f"\nMax facts learned with d_model={d_model}: {max_facts}\n")
#%%

# Save results from above cell
max_facts_d = {16: 32, 32: 64, 64: 256, 128: 384, 256: 512, 512: 1536, 1024: 4096}
#%%

plt.loglog(list(max_facts_d.keys()), list(max_facts_d.values()), marker='o')
plt.xlabel("d_model")
plt.ylabel("Max Facts Learned")
plt.title("Max Facts Learned vs d_model")



xticks = list(max_facts_d.keys())
yticks = [2**i for i in range(4, 13)]

ax = plt.gca()
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels([str(d) for d in xticks])
ax.set_yticklabels([str(d) for d in yticks])
ax.xaxis.set_minor_locator(plt.NullLocator())
ax.yaxis.set_minor_locator(plt.NullLocator())

plt.grid()


plt.show()
#%%
for d in [32*32]:
    settings = generate_settings(d)
    print(f"\nEvaluating hand with n_facts = {settings.n_facts}")

    for top_n in [0, 1]:
        print(f"Evaluating model with top_n = {top_n}")

        settings.top_n = top_n
        model = HandCodedModel(settings)
        accuracy, best_guess_accuracy, logits, hidden = model.evaluate()
        print(f"Accuracy: {accuracy}, Best Guess Accuracy: {best_guess_accuracy}")

# %%
settings = HandCodedModelSettings()
model = HandCodedModel(settings)
for top_n in [0,1,2]:
    settings.top_n = top_n
    model = HandCodedModel(settings)
    plt.figure(figsize=(7, 2))
    plt.imshow(model.up_matrix.cpu(), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.show()
    

# %%

# Visualize the logits
import matplotlib.pyplot as plt

plt.imshow(logits.cpu().detach().numpy(), aspect='auto', cmap='viridis')
plt.colorbar()
plt.xlabel("Output Neurons")
plt.ylabel("Input Examples")
plt.title("Logits Heatmap")
plt.show()

# %%
plt.figure(figsize=(10, 6))
plt.imshow(model.up_matrix.cpu(), aspect='auto', cmap='viridis')
plt.colorbar()
plt.xlabel("Input Neurons")
plt.ylabel("Hidden Neurons")
plt.title("Up Matrix Heatmap")
plt.show()

# %%
plt.figure(figsize=(10, 6))
plt.imshow(model.down_matrix.cpu(), aspect='auto', cmap='viridis')
plt.colorbar()
plt.xlabel("Hidden Neurons")
plt.ylabel("Output Neurons")
plt.title("Down Matrix Heatmap")
plt.show()

# %%


for d in [32]:
    for n_facts in [64, 128, 256, 512, 1024]:
        top_n, acc = search_best_top_n(d, n_facts, metric='best_guess_accuracy')
        print(f"d={d}, n_facts={n_facts}, best guess accuracy={acc:.4f}, best top_n={top_n}")
    


# %%
adjustments = False
results=[]

for acc in [1, 0.9, 0.5]:  

    for d in [16, 32, 64, 128]:
        max_facts, top_n, best_acc = search_max_facts(d, acc, metric='best_guess_accuracy', adjustments=adjustments)
        print(f"d={d}, accuracy_threshold={acc}, max_facts={max_facts}, best_top_n={top_n}, best_accuracy={best_acc:.4f}")

        results.append({'d': d, 'accuracy_threshold': acc, 'max_facts': max_facts, 'best_top_n': top_n, 'best_accuracy': best_acc})

print(results)

# %%


# with adjustments = True
results_at = [{'d': 16, 'accuracy_threshold': 1, 'max_facts': 48, 'best_top_n': 0, 'best_accuracy': 1.0}, 
           {'d': 32, 'accuracy_threshold': 1, 'max_facts': 64, 'best_top_n': 0, 'best_accuracy': 1.0}, 
           {'d': 64, 'accuracy_threshold': 1, 'max_facts': 256, 'best_top_n': 0, 'best_accuracy': 1.0}, 
           {'d': 128, 'accuracy_threshold': 1, 'max_facts': 512, 'best_top_n': 0, 'best_accuracy': 1.0}, 
           {'d': 16, 'accuracy_threshold': 0.9, 'max_facts': 48, 'best_top_n': 0, 'best_accuracy': 1.0}, 
           {'d': 32, 'accuracy_threshold': 0.9, 'max_facts': 128, 'best_top_n': 0, 'best_accuracy': 0.9609375}, 
           {'d': 64, 'accuracy_threshold': 0.9, 'max_facts': 384, 'best_top_n': 0, 'best_accuracy': 0.90625}, 
           {'d': 128, 'accuracy_threshold': 0.9, 'max_facts': 896, 'best_top_n': 0, 'best_accuracy': 0.9375000596046448}, 
           {'d': 16, 'accuracy_threshold': 0.5, 'max_facts': 160, 'best_top_n': 4, 'best_accuracy': 0.5375000238418579}, 
           {'d': 32, 'accuracy_threshold': 0.5, 'max_facts': 480, 'best_top_n': 1, 'best_accuracy': 0.5145833492279053}, 
           {'d': 64, 'accuracy_threshold': 0.5, 'max_facts': 640, 'best_top_n': 1, 'best_accuracy': 0.5062500238418579}, 
           {'d': 128, 'accuracy_threshold': 0.5, 'max_facts': 1280, 'best_top_n': 0, 'best_accuracy': 0.539843738079071}]

# with adjustments = False
results_af = [{'d': 16, 'accuracy_threshold': 1, 'max_facts': 16, 'best_top_n': 0, 'best_accuracy': 1.0}, 
           {'d': 32, 'accuracy_threshold': 1, 'max_facts': 64, 'best_top_n': 0, 'best_accuracy': 1.0}, 
           {'d': 64, 'accuracy_threshold': 1, 'max_facts': 192, 'best_top_n': 0, 'best_accuracy': 1.0}, 
           {'d': 128, 'accuracy_threshold': 1, 'max_facts': 384, 'best_top_n': 0, 'best_accuracy': 1.0}, 
           {'d': 16, 'accuracy_threshold': 0.9, 'max_facts': 48, 'best_top_n': 0, 'best_accuracy': 0.9583333730697632}, 
           {'d': 32, 'accuracy_threshold': 0.9, 'max_facts': 160, 'best_top_n': 0, 'best_accuracy': 0.9312500357627869}, 
           {'d': 64, 'accuracy_threshold': 0.9, 'max_facts': 512, 'best_top_n': 1, 'best_accuracy': 0.91015625}, 
           {'d': 128, 'accuracy_threshold': 0.9, 'max_facts': 1408, 'best_top_n': 1, 'best_accuracy': 0.909801185131073}, 
           {'d': 16, 'accuracy_threshold': 0.5, 'max_facts': 176, 'best_top_n': 2, 'best_accuracy': 0.5056818127632141}, 
           {'d': 32, 'accuracy_threshold': 0.5, 'max_facts': 480, 'best_top_n': 3, 'best_accuracy': 0.5208333730697632}, 
           {'d': 64, 'accuracy_threshold': 0.5, 'max_facts': 1408, 'best_top_n': 2, 'best_accuracy': 0.5007102489471436}, 
           {'d': 128, 'accuracy_threshold': 0.5, 'max_facts': 3840, 'best_top_n': 4, 'best_accuracy': 0.5171875357627869}]

for adjustments in [True, False]:

    if adjustments == True:
        results = results_at
    else:
        results = results_af

    for acc in [1, 0.9, 0.5]:  
        max_facts = [r['max_facts'] for r in results if r['accuracy_threshold'] == acc]
        d_values = [r['d'] for r in results if r['accuracy_threshold'] == acc]
        style = '--x' if adjustments else '-o'
        plt.loglog(d_values, max_facts, style, label=f'Accuracy Threshold = {acc}, adjustments = {adjustments}')
    
plt.xlabel("d_model")
plt.ylabel("Max Facts Learned")
plt.title("Max Facts Learned vs d_model")
plt.legend()

ax = plt.gca()
ax.set_xticks([16, 32, 64, 128])
ax.set_xticks([], minor=True)
ax.set_xticklabels([16, 32, 64, 128])
ax.set_yticks([16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
ax.set_yticks([], minor=True)
ax.set_yticklabels([16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
plt.grid(True, which="both", ls="--", linewidth=0.5)

plt.show()
# %%
