# %%
import torch
from models import generate_facts

#set default device to CPU for this script, since it doesn't require GPU
torch.set_default_device("cpu")

n_facts = 280
input_vocab_size = 32
output_vocab_size = 16
hidden_dim = 14

half_hidden_dim = hidden_dim // 2

facts = generate_facts(n_facts = n_facts,
                       input_len = 2,
                       input_vocab_size = input_vocab_size,
                       output_vocab_size = output_vocab_size,)


first_tokens = facts['inputs'][:,0]
second_tokens = facts['inputs'][:,1]

#turn the token indices into one-hot vectors
first_tokens_one_hot = torch.nn.functional.one_hot(first_tokens, num_classes=input_vocab_size).float()
second_tokens_one_hot = torch.nn.functional.one_hot(second_tokens, num_classes=input_vocab_size).float()

#add the second token at the end of the first token's one-hot vector 
#to create a single input vector
inputs = torch.cat((first_tokens_one_hot, second_tokens_one_hot), dim=1)
if False:
    inputs = torch.einsum('ab, ib -> ia', 
                        torch.rand(input_vocab_size * 2, input_vocab_size * 2), 
                        inputs)  # random linear transformation 

#print(f"facts: {facts}")
#print(f"inputs: {inputs}")



G = torch.rand(half_hidden_dim, input_vocab_size * 2) - 0.4 #slight possitive bias.
Gk = torch.einsum('ab, ib -> ia', G, inputs)
Gk_act = (Gk > 0).float()
Gk_act_k = torch.einsum('ia, ib -> iab', Gk_act, inputs)

Gk_act_k_flat = Gk_act_k.view(n_facts, half_hidden_dim * input_vocab_size * 2)

#print(f"Gk_act_k: {Gk_act_k}")
#print(f"Gk_act_k_flat: {Gk_act_k_flat}")

if False:
    # visualize the Gk_act_k_flat tensor as a heatmap
    import matplotlib.pyplot as plt
    plt.imshow(Gk_act_k_flat, cmap='hot')
    plt.colorbar()
    plt.title("Gk_act_k_flat heatmap")
    plt.xlabel("half_hidden_dim * input_vocab_size * 2")
    plt.ylabel("n_facts")
    plt.show()

# find the rank of the Gk_act_k_flat matrix
rank = torch.linalg.matrix_rank(Gk_act_k_flat)#

# number of rows that are all zeros
num_zero_rows = torch.sum(torch.all(Gk_act_k_flat == 0, dim=1)).item()


print(f"Rank of Gk_act_k_flat: {rank} (out of {n_facts})")
print(f"Number of zero rows in Gk_act_k_flat: {num_zero_rows} (out of {n_facts})")

#Gk_act_k_flat @ A = labels, solve for A using pseudo-inverse
A_flat = torch.linalg.pinv(Gk_act_k_flat) @ facts['targets'].float()

A = A_flat.view(half_hidden_dim, input_vocab_size * 2)

intermediate_outputs = torch.einsum('ia, ac, ic -> i', Gk_act, A, inputs)
intermediate_outputs_correct = (intermediate_outputs.round().long() == facts['targets']).all().item()
print(f"Intermediate outputs correct: {intermediate_outputs_correct}")

output_options = torch.arange(output_vocab_size)
logits = intermediate_outputs[:,None] * output_options[None,:] - output_options[None,:]**2 / 2
predictions = torch.argmax(logits, dim=1)
predictions_correct = (predictions == facts['targets']).all().item() 
print(f"Predictions correct: {predictions_correct}")

Ak = torch.einsum('ab, ib -> ia', A, inputs)

l = (Ak/Gk).min(dim=0).values
L = torch.diag(l)

#%%

#The scaling from the paper dosn't work. Probably I did someting wrong.
#Just scaling by 10_000 instead works. 
LG = 100_000 *G  #L@G 

W_up = torch.cat((LG + A, LG), dim=0)

hidden = torch.relu(inputs @ W_up.T)

adding_up = torch.cat([torch.ones(half_hidden_dim), -torch.ones(half_hidden_dim)])
intermediate_outputs2 = hidden @ adding_up
print(f"Intermediate outputs2: {intermediate_outputs2.round().long()}")
print(f"intermediate outputs2 correct: {(intermediate_outputs2.round().long() == facts['targets']).all().item()}    ")


W_down = torch.einsum('a,b -> ab', output_options**2, adding_up) 
b_down = -2/3 * output_options**3

logits2 = hidden @ W_down.T + b_down
outputs2 = torch.argmax(logits2, dim=1)
print(f"Outputs2: {outputs2}")

outputs2_correct = (outputs2 == facts['targets']).all().item()
print(f"Outputs2 correct: {outputs2_correct}")

# %%
plt.plot(intermediate_outputs2.detach().numpy(), label='Intermediate Outputs')
plt.legend()
plt.show()
# %%
plt.plot(logits2[100:].detach().numpy(), label='Logits')

plt.show()
# %%
plt.plot(outputs2.detach().numpy(), label='Outputs')
plt.legend()
plt.show()
# %%
