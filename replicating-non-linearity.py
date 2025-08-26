import torch
import matplotlib.pyplot as plt


A = torch.arange(-10, 10, 1, dtype=torch.float32)

# plt.plot(A)
# plt.plot(torch.relu(A))
def myRelu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.tensor(0), x)

def mySigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))

results = myRelu(A)
resultsB = mySigmoid(A)
# plt.plot(results)
plt.plot(resultsB)
plt.show()