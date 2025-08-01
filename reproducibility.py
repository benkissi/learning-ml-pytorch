## Reproducibility - trying to take randomness out of the random
## In short how a neural network learns:

# `start with random numbers -> tensor operations -> update random numbers to try and make them better representations
# of the data -> repeat until satisfied`

# To reduce the randomness, we can set a random seed.
# Essentially, a random seed is a starting point for the random number generator.

import torch

random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

# print(random_tensor_A)
# print(random_tensor_B)

# print(random_tensor_A == random_tensor_B)

#lets mae some random tensors reproducible

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)

print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)