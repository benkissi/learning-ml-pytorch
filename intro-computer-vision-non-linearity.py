import torch
from torch import nn

import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tqdm.auto import tqdm

from helper_functions import plot_decision_boundary, accuracy_fn, print_train_time, train_step, test_step, eval_model

# Getting dataset
# Using MNIST dataset - FASHION MNIST
train_data = datasets.FashionMNIST(
    root="data", #where to store the data
    train=True, # do we want the training dataset
    download=True, # do we want to download the dataset
    transform=ToTensor(), # how do we want to transform the data
    target_transform=None # we don't need to transform the labels
)

test_data = datasets.FashionMNIST(
    root="data", #where to store the data
    train=False, # do we want the test dataset
    download=True, # do we want to download the dataset
    transform=ToTensor(), # how do we want to transform the data
    target_transform=None # we don't need to transform the labels
)


# prepare dataloader
BATCH_SIZE = 32
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
class_names = train_data.classes

# device = "cpu" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),  # flatten the input into a single vector
            nn.Linear(in_features=input_shape, out_features=hidden_units),  # hidden layer
            nn.ReLU(),  # non-linearity
            nn.Linear(in_features=hidden_units, out_features=output_shape),  # output layer
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)

torch.manual_seed(42)

model_1 = FashionMNISTModelV1(
    input_shape=784,
    hidden_units=128,
    output_shape=len(class_names)
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)

train_time_start = timer()

# set number of epochs
epochs = 3

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_step(model_1, train_dataloader, loss_fn, optimizer, accuracy_fn, device)
    test_step(model_1, test_dataloader, loss_fn, accuracy_fn, device)

train_time_end = timer()
print_train_time(train_time_start, train_time_end, device)


model_results = eval_model(model_1, test_dataloader, loss_fn, accuracy_fn, device)
print(f"Model results: {model_results}")
