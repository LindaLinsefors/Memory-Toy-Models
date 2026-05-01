# %%

# We want to solve for V and E, given any x1, x2, E1, E2

# R = E1 x1 + E2 x2

# R = E x1 + 1/2 V E x1 + 1/2 V E x2
# R = (E + 1/2 V E) x1 + 1/2 V E x2

# E = E1 - E2
# %%


# %%


E1 = torch.rand(4,8)
E2 = torch.rand(4,8)

E = E1 - E2
V = 2 * E2 @ torch.linalg.pinv(E)

#check
print(E1 - (E + 1/2 * V @ E))
print(E2 - (1/2 * V @ E))

import torch
# %%
M = torch.tensor([[1., 2., 0.],
                  [0., 1., 1.]]) 
M = M.T
N = torch.linalg.pinv(M)

M @ N
# %%
