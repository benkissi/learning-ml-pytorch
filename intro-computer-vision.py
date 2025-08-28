import torch
from torch import nn

import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tqdm.auto import tqdm

from helper_functions import plot_decision_boundary, accuracy_fn, print_train_time, eval_model

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

device = "cpu" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(device)
torch.manual_seed(42)

model_0 = FashionMNISTModelV0(
    input_shape=784,
    hidden_units=10,
    output_shape=len(class_names)
).to(device)

# dummy_x = torch.randn([1, 1, 28, 28])
# print(dummy_x.shape)

# dummy_output = model_0(dummy_x)
# print(dummy_output.shape)

# Setup loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

# 1. Loop through epochs
# 2. Loop through training batches, perform training steps, calculate the train loss per batch
# 3. Loop through testing batches, perform testing steps, calculate the test loss per batch
# 4. Print out whats happening
# 5. Time it all for fun

# training loop
torch.manual_seed(42)
train_time_start = timer()

# set number of epochs
epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    # training
    train_loss = 0

    # add a loop to loop through the training batches
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        model_0.train()

        #1. Forward pass
        y_pred = model_0(X)

        #2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        #3. Optimizer zero grad
        optimizer.zero_grad()

        #4. Loss backward
        loss.backward()

        #5. optimizer step
        optimizer.step()
        
        # print out whats happening
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")
    
    # divide total train loss by length of train dataloader to get the average loss per batch
    train_loss = train_loss / len(train_dataloader)

    # Testing
    test_loss, test_acc = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            #1. Forward pass
            test_pred = model_0(X_test)

            #2. Calculate loss
            loss = loss_fn(test_pred, y_test)
            test_loss += loss.item()

            #3. Calculate accuracy
            test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))

        # calculate average loss and accuracy per batch
        test_loss = test_loss / len(test_dataloader)
        test_acc = test_acc / len(test_dataloader)
    
    # Print out whats happening
    print(f"Train loss: {train_loss:.4f}")
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

train_time_end = timer()
total_train_time_model_0 = print_train_time(start=train_time_start, end=train_time_end, device=device)



model_0_results = eval_model(
    model=model_0,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn
)

print(model_0_results)
