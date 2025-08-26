import torch
from torch import nn
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from helper_functions import accuracy_fn, plot_decision_boundary

from torchmetrics import Accuracy

# Set the hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
NUM_SAMPLES = 1000
RANDOM_SEED = 42

# Create multi-class data
X_blob, y_blob = make_blobs(
    n_samples=NUM_SAMPLES, 
    n_features=NUM_FEATURES, 
    centers=NUM_CLASSES, 
    cluster_std=1.5, #give the clusters a little shake up
    random_state=RANDOM_SEED
    )

# Convert to PyTorch tensors
X_blob = torch.from_numpy(X_blob).type(torch.float32)
y_blob = torch.from_numpy(y_blob).type(torch.float32)

# Split into train and test sets
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(
    X_blob,
    y_blob,
    test_size=0.2,
    random_state=RANDOM_SEED
)

# Plot data
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap="viridis", s=100, alpha=0.7)
plt.title("Multi-class Classification Data")
# plt.show()


# device agnostic code
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Building multi-class classification model
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        """Initializes multi-class classification model"
        
        Args:
            input_features (int): Number of input features.
            output_features (int): Number of output classes.
            hidden_units (int, optional): Number of hidden units in the hidden layer. Defaults to 8.
        """
        super().__init__()

        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)

print(X_blob.shape, y_blob.shape)
# Creat an instance of Blob model and send it target device
model_0 = BlobModel(input_features=2, output_features=4).to(device)

# print(model_0.state_dict())

# create loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

torch.manual_seed(RANDOM_SEED)
torch.mps.manual_seed(RANDOM_SEED)

X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)
X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)

epochs = 100
epoch_count = []
loss_values = []
test_loss_values = []
acc_test_values = []

# In order to evaluate and train and test the model, we need to convert the models outputs (logits) into prediction
# probabilities and then to prediction labels

# Logits -> Pred probs -> Pred Labels
# Get raw logits from model and then convert them to probabilities using softmax
# then convert to prediction labels using argmax

# Training loop
for epoch in range(epochs):
    model_0.train()

    y_logits = model_0(X_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    # Calculate loss
    loss = loss_fn(y_logits, y_blob_train)

    # Calculate accuracy
    acc = accuracy_fn(y_true=y_blob_train, y_pred=y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        y_logits_test = model_0(X_blob_test)
        loss_test = loss_fn(y_logits_test, y_blob_test)
        y_pred_test = torch.softmax(y_logits_test, dim=1).argmax(dim=1)

        acc_test = accuracy_fn(y_true=y_blob_test, y_pred=y_pred_test)
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            loss_values.append(loss.item())
            test_loss_values.append(loss_test.item())
            acc_test_values.append(acc_test)
            
            print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test Loss: {loss_test:.5f} | Acc: {acc:.2f} | Test Acc: {acc_test:.2f}")

# Plotting the decision boundary
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Training Data Decision Boundary")
plot_decision_boundary(model_0, X_blob_train, y_blob_train)

plt.subplot(1, 2, 2)
plt.title("Test Data Decision Boundary")
plot_decision_boundary(model_0, X_blob_test, y_blob_test)
# plt.show()

torchmetric_accuracy = Accuracy(task="multiclass", num_classes=4).to(device)

accuracy = torchmetric_accuracy(y_pred_test, y_blob_test)

print(f"Torch Metric Accuracy: {accuracy}")


# Classification metrics
# Accuracy - out of 100 samples, how many does our model get right
# Precision - out of all positive predictions, how many were actually positive?
# Recall - out of all actual positives, how many did we predict correctly?
# F1 Score - Combines precision and recall into a single metric
# Confusion Matrix - A table to visualize the performance of a classification model
# Classification Report - A summary of precision, recall, F1 score, and support for each class