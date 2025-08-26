from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch

from helper_functions import  plot_decision_boundary 

n_samples = 1000

# create circles
X, y = make_circles(n_samples=n_samples, noise=0.03, random_state=42)

print(len(X), len(y))

print(X[:5])
print(y[:5])

circles = pd.DataFrame({
    'x1': X[:, 0],
    'x2': X[:, 1],
    'label': y
})

# print(circles.head(10))

# plt.scatter(
#     x=X[:, 0],
#     y=X[:, 1],
#     c=y,
#     cmap=plt.cm.RdYlBu
# )
# plt.show()

print(X.shape, y.shape)

# convert data into tensors
X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

# split data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(len(X_train), len(y_train), len(X_test), len(y_test))


device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x)) # x -> layer_1 -> layer_2 -> output

model_0 = CircleModelV0().to(device)
print(f'model: {next(model_0.parameters()).device}')

# lets replicate the model above with nn.Sequential
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)
print(f'model: {next(model_0.parameters()).device}')
print(model_0.state_dict())

with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))
print(f"Length of predicitions: {len(untrained_preds)}, shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(X_test)}, shape: {X_test.shape}")
print(f"\nFirst 10 predictions:\n {untrained_preds[:10]}")
print(f"\nFirst 10 labels:\n {y_test[:10]}")

#setup loss function and optimizer
# which loss function or optimizer to use?
# Regression (predicting a number) -> MSELoss, MAELoss
# Classification (predicting a 0 or 1) -> BinaryCrossEntropyLoss or CategoricalCrossEntropyLoss
# The loss function measures how wrong the model's predictions are
# for optimizer, we can use SGD, Adam, however pytorch has many built in optimizers

loss_fn = nn.BCEWithLogitsLoss() # Binary Cross Entropy with Logits Loss = sigmoid activation function built in

optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1) # Stochastic Gradient Descent with learning rate of 0.1

#calculate accuracy function - out of 100 examples, what percentage does the model get right?
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = correct / len(y_pred) * 100
    return acc
# Raw logits -> Predicted probabilities -> Prediction labels
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))
# print(f"Logits--------: {y_logits[:10]}")

y_pred_probs = torch.sigmoid(y_logits)
# print(f"Predicted probabilities---------: {y_pred_probs[:10]}")

# Find the predicted labels
y_preds = torch.round(y_pred_probs)
# print(f"Predictions (rounded probabilities)---------: {y_preds[:10]}")

y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

print(f"Predicted labels---------:", torch.eq(y_preds.squeeze()[:5], y_pred_labels.squeeze()))
print(y_preds.squeeze()[:5])

torch.mps.manual_seed(42)
torch.manual_seed(42)

epochs = 1000
epoch_count = []
loss_values = []
test_loss_values = []
test_acc_values = []

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    model_0.train()

    #1. Forward pass
    y_logits = model_0(X_train).squeeze()  # forward pass
    y_preds = torch.round(torch.sigmoid(y_logits))

    #2. Calculate loss and accuracy
    loss = loss_fn(y_logits, y_train) # nn.BCEWithLogitsLoss() expects raw logits
    acc = accuracy_fn(y_true=y_train, y_pred=y_preds)

    #3. Optimizer zero grad
    optimizer.zero_grad()

    #4. loss backward (backpropagation) -> calculate gradients with respect all the parameters in the model
    loss.backward()

    #5. optimizer step (gradient descent) -> update the parameters to reduce the loss
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_preds)
    
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss.item())
        test_loss_values.append(test_loss.item())
        test_acc_values.append(test_acc)
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}%")


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
plt.show()