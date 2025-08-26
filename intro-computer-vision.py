import torch
from torch import nn

import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from helper_functions import plot_decision_boundary, accuracy_fn

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

# print(len(train_data), len(test_data))
image, label = train_data[0]
print(image.shape, label)

#get classes
class_names = train_data.classes
print(class_names)

# class to idx
class_to_idx = train_data.class_to_idx
print(class_to_idx)

# targets
targets = train_data.targets
print(targets)

#check shape
print(f"Image shape: {image.shape} -> [color_channels, height, width]")
print(f"Image label: {class_names[label]}")

# Visualize
# plt.imshow(image.squeeze(), cmap="gray")
# plt.title(class_names[label])
# plt.show()

# plot more images
torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4

# for i in range(1, rows * cols + 1):
#     random_idx = torch.randint(0, len(train_data), size=[1]).item()
#     img, label = train_data[random_idx]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(img.squeeze(), cmap="gray")
#     plt.title(class_names[label])
#     plt.axis(False)

# plt.show()

# prepare dataloader
BATCH_SIZE = 32
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

print(f"Length of train_dataloader: {len(train_dataloader)}")
print(f"Length of test_dataloader: {len(test_dataloader)}")

train_features_batch, train_labels_batch = next(iter(train_dataloader))

print(train_features_batch.shape, train_labels_batch.shape)

# torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]

plt.title(class_names[label])
plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

# When building a series of machine learning modelling experiments, its good practise to start with a 
# baseline model.

#create flatten layer
flatten_model = nn.Flatten()

x = train_features_batch[0]

output = flatten_model(x)

print(f"shape after flatten: {output.shape}")

class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),  # flatten the input
            nn.Linear(in_features=input_shape, out_features=hidden_units),  # hidden layer
            nn.Linear(in_features=hidden_units, out_features=output_shape)  # output layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)
    

torch.manual_seed(42)

model_0 = FashionMNISTModelV0(
    input_shape=784,
    hidden_units=10,
    output_shape=len(class_names)
)

dummy_x = torch.randn([1, 1, 28, 28])
print(dummy_x.shape)

dummy_output = model_0(dummy_x)
print(dummy_output.shape)
