import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

#known parameters in linear regression
weight = 0.7
bias = 0.3

#create
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(1)
y = weight * X + bias

# Splitting data into training and testing sets

train_split = int(0.8 * len(X))
# print("Train split index:", train_split)

X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# print(len(X_train), len(y_train))
# print(len(X_test), len(y_test))

def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None):
    """Plot training and testing data with predictions if provided."""
    plt.figure(figsize=(10, 7))

    plt.scatter(train_data, train_labels, c="b", label="Training Data")
    plt.scatter(test_data, test_labels, c="g", label="Testing Data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()

## create a linear regression model class
class LinearRegressionModel(nn.Module): # <- almost eveything in PyTorch inherits from nn.Module
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

        #forarward method to define computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.weights * x + self.bias
        
torch.manual_seed(42)

model_0 = LinearRegressionModel()

# print(list(model_0.parameters()))
# print(model_0.state_dict())

# # make predictions
# with torch.inference_mode():
#     y_preds = model_0(X_test)

# print(y_preds)



#To train the model, we need to define a loss function and an optimizer
# Loss function: Mean Absolute Error (MAE) -> A function to measure how far off our predictions are from the ground truth
# Optimizer: Stochastic Gradient Descent (SGD) -> A function to update the model parameters based on the loss
# For pytorch we need a training loop and a testing loop

#setup loss function
loss_fn = nn.L1Loss()

#setup optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

## Building a traing and test loop
# 1. Loop through the data
# 2. Forward pass: compute predictions also known as forward propagation
# 3. Calculate the loss
# 4. Optimizer Zero the gradients
# 5. Loss backward: move backwards through the model to calculate the gradients of each parameter of the model with respect to the loss (back propagation)
# 6. Optimizer step: update the parameters of the model using the gradients calculated in the previous step to improve the loss (gradient descent)

torch.manual_seed(42)

# An epoch is one complete pass through the dataset.
epochs = 200
epoch_count = []
loss_values = []
test_loss_values = []
#loop through the data
for epoch in range(epochs):
    model_0.train()  # set the model to training mode which sets all parameters that require gradients to True
    # 1. forward pass
    y_pred = model_0(X_train)

    # 2. calculate the loss
    loss = loss_fn(y_pred, y_train)
    print(f"Loss: {loss}")

    # 3. optimizer zero the gradients
    optimizer.zero_grad()
    # 4. loss backward
    loss.backward()
    # 5. optimizer step (perform gradient descent)
    optimizer.step() # by default optimizer changes accumulates in the loop so it needs to be zeroed out at the start of each loop

    #Testing the model
    model_0.eval() #turns off different settings in the model not needed to evaluation/testing
    with torch.inference_mode(): # turns of gradient tracking and other settings not needed for inference
        #1. forward pass
        test_pred = model_0(X_test)

        #2. calculate the loss
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)

        print(f"Epoch: {epoch} | Train Loss: {loss:.5f} | Test Loss: {test_loss:.5f}")
        print(model_0.state_dict())  # returns the state of the model parameters

#Plot loss curves


def plot_loss_curves(epoch_count, loss_values, test_loss_values):
    """Plot training and testing loss curves."""
    plt.plot(epoch_count, loss_values, label="Train Loss")
    plt.plot(epoch_count, test_loss_values, label="Test Loss")

    plt.title("Training and test loss curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

plot_loss_curves(epoch_count, np.array(torch.tensor(loss_values).numpy()), test_loss_values)

with torch.inference_mode():
    y_preds_new = model_0(X_test)

# Plot the predictions
plot_predictions(predictions=y_preds_new)

