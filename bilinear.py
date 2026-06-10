#%%
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_normalize

# Use:
# X: shape (I, J, K)
# weights, factors = parafac(X, rank=R, n_iter_max=500, init="svd", tol=1e-7)
# factors = [A, B, C] where A:(I,R), B:(J,R), C:(K,R)


import torch
import torch.nn.functional as F
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


#%%
for n_facts in [16, 32, 64, 128, 256, 512, 1024]:
    print()
    for rank in [2, 4, 8, 16, 32, 64, 128, 256, 512]:

        seq_len = 2
        input_vocab_size = 32
        output_vocab_size = 16

        facts = generate_facts(n_facts, seq_len, input_vocab_size, output_vocab_size)

        x1 = F.one_hot(facts["inputs"][:, 0], num_classes=input_vocab_size).float()
        x2 = F.one_hot(facts["inputs"][:, 1], num_classes=input_vocab_size).float()
        y =  F.one_hot(facts["targets"], num_classes=output_vocab_size).float()

        fact_tensor = torch.einsum("ni, nj, nk -> i j k", x1, x2, y)

        tl.set_backend("pytorch")
        cp = parafac(
            fact_tensor.cpu(),
            rank=rank,
            n_iter_max=2000,
            init="svd",
            tol=1e-10,
            l2_reg=1e-8,
            linesearch=True,
        )

        # Normalize after the (stable) unnormalized solve to recover meaningful weights.
        weights, factors = cp_normalize(cp)

        new_fact_tensor = tl.cp_to_tensor((weights, factors)).to("cuda")

        logits = torch.einsum("ni,nj,ijk -> nk", x1, x2, new_fact_tensor)
        accuracy = (logits.argmax(dim=-1) == facts["targets"]).float().mean()
        print(f"n_facts: {n_facts}, rank: {rank}, accuracy: {accuracy.item()}")
        print(weights)


#%%
for n_facts in [256]:
    print()
    for rank in [2, 4, 8, 16, 32, 64, 128, 256, 512]:

        seq_len = 2
        input_vocab_size = 16
        output_vocab_size = 16

        facts = generate_facts(n_facts, seq_len, input_vocab_size, output_vocab_size)

        x1 = F.one_hot(facts["inputs"][:, 0], num_classes=input_vocab_size).float()
        x2 = F.one_hot(facts["inputs"][:, 1], num_classes=input_vocab_size).float()
        y =  F.one_hot(facts["targets"], num_classes=output_vocab_size).float()

        fact_tensor = torch.einsum("ni, nj, nk -> i j k", x1, x2, y)

        tl.set_backend("pytorch")
        cp = parafac(
            fact_tensor.cpu(),
            rank=rank,
            n_iter_max=2000,
            init="svd",
            tol=1e-10,
            l2_reg=1e-8,
            linesearch=True,
        )

        # Normalize after the (stable) unnormalized solve to recover meaningful weights.
        weights, factors = cp_normalize(cp)

        new_fact_tensor = tl.cp_to_tensor((weights, factors)).to("cuda")

        logits = torch.einsum("ni,nj,ijk -> nk", x1, x2, new_fact_tensor)
        accuracy = (logits.argmax(dim=-1) == facts["targets"]).float().mean()
        print(f"n_facts: {n_facts}, rank: {rank}, accuracy: {accuracy.item()}")
        print(weights)


# %%
for n_facts in [256]:
    print()
    for rank in [32]:

        seq_len = 2
        input_vocab_size = 16
        output_vocab_size = 16

        facts = generate_facts(n_facts, seq_len, input_vocab_size, output_vocab_size)

        x1 = F.one_hot(facts["inputs"][:, 0], num_classes=input_vocab_size).float()
        x2 = F.one_hot(facts["inputs"][:, 1], num_classes=input_vocab_size).float()
        y =  F.one_hot(facts["targets"], num_classes=output_vocab_size).float()

        fact_tensor = torch.einsum("ni, nj, nk -> i j k", x1, x2, y)

        tl.set_backend("pytorch")
        cp = parafac(
            fact_tensor.cpu(),
            rank=rank,
            n_iter_max=2000,
            init="svd",
            tol=1e-10,
            l2_reg=1e-8,
            linesearch=True,
        )

        # Normalize after the (stable) unnormalized solve to recover meaningful weights.
        weights, factors = cp_normalize(cp)

        new_fact_tensor = tl.cp_to_tensor((weights, factors)).to("cuda")

        logits = torch.einsum("ni,nj,ijk -> nk", x1, x2, new_fact_tensor)
        accuracy = (logits.argmax(dim=-1) == facts["targets"]).float().mean()
        print(f"n_facts: {n_facts}, rank: {rank}, accuracy: {accuracy.item()}")
        print(weights)
# %%
factors1, factors2, factors3 = factors

# vissualise factors as a matrix of shape (input_vocab_size, rank) for factors1 and factors2, and (output_vocab_size, rank) for factors3
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 5))
plt.subplot(1, 3, 1)
plt.imshow(factors1.cpu().detach().numpy(), aspect='auto')
plt.title("Factors 1 (input 1)")
plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(factors2.cpu().detach().numpy(), aspect='auto')
plt.title("Factors 2 (input 2)")
plt.colorbar()
plt.subplot(1, 3, 3)
plt.imshow(factors3.cpu().detach().numpy(), aspect='auto')
plt.title("Factors 3 (output)")
plt.colorbar()
plt.show()  



# %%
