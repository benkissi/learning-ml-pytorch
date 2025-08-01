import torch
import matplotlib.pyplot as plt
from torch import nn
import numpy as np

from pathlib import Path

weights = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(1)
print('X', X.shape)
y = weights * X + bias

train_split = int(0.8 * len(X))

x_train, y_train = X[:train_split], y[:train_split]
x_test, y_test = X[train_split:], y[train_split:]

def plot_predictions(train_data=x_train, train_labels=y_train, test_data=x_test, test_labels=y_test, predictions=None):
    plt.figure(figsize=(10,7))

    plt.scatter(train_data, train_labels, c="b", label="Training data")
    plt.scatter(test_data, test_labels, c="g", label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", label="Predictions")
    
    plt.legend(prop={'size':14})
    plt.show()

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

torch.manual_seed(42)
model_0 = LinearRegressionModel()

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)


torch.manual_seed(42)
epochs = 200
epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):
    # T, P, L, Z, B, O, E
    model_0.train()

    y_pred = model_0(x_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        y_test_pred = model_0(x_test)
        test_loss = loss_fn(y_test_pred, y_test)

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        
        print(f"Epoch: {epoch}, Train Loss: {loss}, Test Loss: {test_loss}")
        print(model_0.state_dict())

def plot_loss_curves(epoch_count, loss_values, test_loss_values):
    plt.plot(epoch_count, loss_values, label="Train loss")
    plt.plot(epoch_count, test_loss_values, label="Test loss")

    plt.title('Training and test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


plot_loss_curves(epoch_count, np.array(
    torch.tensor(loss_values).numpy()), test_loss_values)

with torch.inference_mode():
    new_predictions = model_0(x_test)

plot_predictions(predictions=new_predictions)

# Saving and loading a model
# 1. torch.save() -  allows you to save a pytorch object in Python's pickle format
# 2. torch.load() - allows you to load a saved Pytorch object
# 3. torch.nn.Module.load_state_dict() - allows you load a models saved state dictionary

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "01_pytoch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)
print(f"model_0 {model_0.state_dict()}")

loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(torch.load(MODEL_SAVE_PATH))
print(f"loaded_model_0 {loaded_model_0.state_dict()}")

