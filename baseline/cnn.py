# cnn_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test

# Reshape for CNN: (samples, channels=1, features)
X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32)

train_ds = TensorDataset(X_train_scaled, y_train)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

class_weights = compute_class_weight(
    "balanced", classes=np.unique(y_train), y=y_train.numpy()
)
weights = torch.tensor(class_weights, dtype=torch.float32)


# CNN Model
class CNN1D(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear((input_dim // 2) * 64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return self.sigmoid(x)


model = CNN1D(X_train_scaled.shape[2])
criterion = nn.BCELoss(weight=weights[1])
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb).squeeze()
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    y_proba = model(X_test_scaled).squeeze().numpy()
    y_pred = (y_proba >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

"""  
Accuracy: 0.8544721647238261
ROC-AUC: 0.8154075196913921
Classification Report:
               precision    recall  f1-score   support

         0.0       0.87      0.98      0.92     38922
         1.0       0.58      0.15      0.24      6973

    accuracy                           0.85     45895
   macro avg       0.72      0.56      0.58     45895
weighted avg       0.82      0.85      0.82     45895

Confusion Matrix:
 [[38181   741]
 [ 5938  1035]]
"""
