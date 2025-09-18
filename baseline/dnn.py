import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test, X


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).to(device)


train_size = int(0.8 * len(X_train_scaled))
val_size = len(X_train_scaled) - train_size
train_ds, val_ds = random_split(
    TensorDataset(X_train_scaled, y_train), [train_size, val_size]
)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)


class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train.numpy()), y=y_train.numpy()
)
pos_weight = torch.tensor([class_weights[1]], dtype=torch.float32).to(device)



class DNNModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


model = DNNModel(X_train_scaled.shape[1]).to(device)


criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # smaller LR


epochs = 80
best_val_loss = float("inf")
patience, patience_counter = 5, 0

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb).squeeze()
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


model.load_state_dict(best_model_state)


model.eval()
with torch.no_grad():
    y_pred_logits = model(X_test_scaled).squeeze()
    y_pred_probs = torch.sigmoid(y_pred_logits).cpu().numpy()
    y_pred = (y_pred_probs >= 0.5).astype(int)


y_test_np = y_test.cpu().numpy()

roc_auc = roc_auc_score(y_test_np, y_pred_probs)
print(f"Accuracy: {accuracy_score(y_test_np, y_pred)}")
print(f"ROC AUC: {roc_auc}")
print(f"Classification Report:\n{classification_report(y_test_np, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test_np, y_pred)}")
