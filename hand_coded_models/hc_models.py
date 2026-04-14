# %%
import torch
import torch.nn.functional as F
torch.set_default_device("cuda")

import matplotlib.pyplot as plt

from models import generate_facts
from hand_coded_models.neuron_assigments import generate_neuron_assignments



class HandCodedModelSettings:
    def __init__(self, input_vocab_size=32, output_vocab_size=16, n_facts=16, seed=42,
                 d_ff=16, n_neruons_per_label=3, 
                 use_top_no_top_fraction='top_n', top_n=0, top_fraction=0.2):
        
        # Data dimensions
        self.seq_len = 2  # Fixed for this model
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.n_facts = n_facts
        self.seed = seed  # For reproducibility

        # Internal model settings
        self.d_ff = d_ff
        self.n_neruons_per_label = n_neruons_per_label
        self.use_top_no_top_fraction = use_top_no_top_fraction  # 'top_n' or 'top_fraction'
        self.top_n = top_n
        self.top_fraction = top_fraction

def generate_settings(d):
    return HandCodedModelSettings(input_vocab_size = 2*d, 
                                  output_vocab_size = d, 
                                  n_facts = d,
                                  d_ff = d)

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
        S = settings.n_neruons_per_label
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

            if settings.use_top_no_top_fraction == 'top_fraction':
                top_first_inputs = unique_f[torch.argsort(counts_f, descending=True)[:int(len(unique_f)*settings.top_fraction)]]
                top_second_inputs = unique_s[torch.argsort(counts_s, descending=True)[:int(len(unique_s)*settings.top_fraction)]]

            if settings.use_top_no_top_fraction == 'top_n':
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
        return accuracy, logits, hidden




settings = HandCodedModelSettings()
settings.n_facts = 32
model = HandCodedModel(settings)
accuracy, logits, hidden = model.evaluate()

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
            accuracy, logits, hidden = model.evaluate()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                if best_accuracy == 1.0:
                    break
        print(f"Best Accuracy: {best_accuracy}")
#%%
for d in [32*32]:
    settings = generate_settings(d)
    print(f"\nEvaluating hand with n_facts = {settings.n_facts}")

    for top_n in [0, 1]:
        print(f"Evaluating model with top_n = {top_n}")

        settings.top_n = top_n
        model = HandCodedModel(settings)
        accuracy, logits, hidden = model.evaluate()
        print(f"Accuracy: {accuracy}")

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
