# rnn_model.py
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

# Reshape for RNN: (samples, timesteps=features, input_dim=1)
X_train = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(-1)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32)

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

class_weights = compute_class_weight(
    "balanced", classes=np.unique(y_train), y=y_train.numpy()
)
weights = torch.tensor(class_weights, dtype=torch.float32)


# RNN Model
class RNN_LSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return self.sigmoid(out)


model = RNN_LSTM(X_train.shape[1])
criterion = nn.BCELoss(weight=weights[1])
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 20
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
    y_proba = model(X_test).squeeze().numpy()
    y_pred = (y_proba >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
"""  
Accuracy: 0.8534045102952391
ROC-AUC: 0.8149930458054522
Classification Report:
               precision    recall  f1-score   support

         0.0       0.86      0.99      0.92     38922
         1.0       0.67      0.07      0.13      6973

    accuracy                           0.85     45895
   macro avg       0.76      0.53      0.52     45895
weighted avg       0.83      0.85      0.80     45895

Confusion Matrix:
 [[38686   236]
 [ 6492   481]]
"""
