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
from sklearn.model_selection import StratifiedKFold
import numpy as np
from src.metrics import print_core_metrics

from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test
import optuna as op
 

# Convert labels to numpy
y_train_np = y_train.to_numpy() if hasattr(y_train, 'to_numpy') else np.asarray(y_train)
y_test_np = y_test.to_numpy() if hasattr(y_test, 'to_numpy') else np.asarray(y_test)

# GPU acceleration setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Reshape for CNN: (samples, channels=1, features)
# y_train_np / y_test_np were created above and hold numpy label arrays

# Build tensors then normalize to 3D shape expected by Conv1d: (batch, channels, length)
X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)


def _ensure_3d_tensor(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has shape (batch, channels, length).
    If tensor is 2D (batch, length) -> unsqueeze channel dim.
    If tensor has extra singleton dims (e.g. batch,1,1,length) -> squeeze them.
    """
    # Move any extra singleton dims (except batch dim) out
    while x.dim() > 3:
        # find a singleton dim other than batch axis
        for d in range(1, x.dim()):
            if x.size(d) == 1:
                x = x.squeeze(d)
                break
        else:
            break
    if x.dim() == 2:
        x = x.unsqueeze(1)
    return x


X_train_scaled = _ensure_3d_tensor(X_train_scaled)
X_test_scaled = _ensure_3d_tensor(X_test_scaled)

# Create torch tensors for labels once
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32).to(device)

train_ds = TensorDataset(X_train_scaled, y_train_tensor)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

class_weights = compute_class_weight(
    "balanced", classes=np.unique(y_train_np), y=y_train_np
)
weights = torch.tensor(class_weights, dtype=torch.float32).to(device)


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


def train_and_eval(hparams: dict):
    model = CNN1D(X_train_scaled.shape[2]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=hparams.get('weight_decay', 0.0))
    criterion = nn.BCELoss()

    epochs = hparams.get('epochs', 30)
    batch_size = hparams.get('batch_size', 256)

    # X_train_scaled is already a tensor shaped (n_samples, 1, n_features) so don't unsqueeze again
    train_ds = TensorDataset(X_train_scaled, y_train_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        # X_test_scaled is already a tensor on device with shape (n_samples, 1, n_features)
        X_test_t = X_test_scaled
        y_proba = model(X_test_t).squeeze().cpu().numpy()
        y_pred = (y_proba >= 0.5).astype(int)

    # Print core metrics + detailed report for test set
    print("\n=== Test set metrics ===")
    print_core_metrics(y_test_np, y_pred)
    print("Classification Report:\n", classification_report(y_test_np, y_pred))

    # Cross-validation on training set (Stratified 5-fold)
    print("\n=== 5-fold cross-validation on training set ===")
    X_cv = X_train_scaled.cpu().numpy()
    # collapse channel dim if present: (n,1,L) -> (n,L)
    if X_cv.ndim == 3 and X_cv.shape[1] == 1:
        X_cv_flat = X_cv.reshape(X_cv.shape[0], X_cv.shape[2])
    else:
        X_cv_flat = X_cv.reshape(X_cv.shape[0], -1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    fold_accs = []
    L = X_train_scaled.shape[2]
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_cv_flat, y_train_np), 1):
        # prepare fold tensors
        X_tr = torch.tensor(X_cv_flat[tr_idx], dtype=torch.float32).to(device).reshape(-1, 1, L)
        y_tr = torch.tensor(y_train_np[tr_idx], dtype=torch.float32).to(device)
        X_val = torch.tensor(X_cv_flat[val_idx], dtype=torch.float32).to(device).reshape(-1, 1, L)
        y_val = torch.tensor(y_train_np[val_idx], dtype=torch.float32).to(device)

        fold_model = CNN1D(L).to(device)
        opt = optim.Adam(fold_model.parameters(), lr=hparams.get('lr', 1e-3))
        crit = nn.BCELoss()
        cv_epochs = min(10, epochs)
        batch = min(128, max(32, len(tr_idx)//10))
        ds = TensorDataset(X_tr, y_tr)
        loader = DataLoader(ds, batch_size=batch, shuffle=True)
        for e in range(cv_epochs):
            fold_model.train()
            for xb, yb in loader:
                opt.zero_grad()
                preds = fold_model(xb).squeeze()
                loss = crit(preds, yb)
                loss.backward()
                opt.step()

        fold_model.eval()
        with torch.no_grad():
            y_val_proba = fold_model(X_val).squeeze().cpu().numpy()
            y_val_pred = (y_val_proba >= 0.5).astype(int)
        fold_aucs.append(roc_auc_score(y_val.cpu().numpy(), y_val_proba))
        fold_accs.append(accuracy_score(y_val.cpu().numpy(), y_val_pred))
        print(f"Fold {fold}: AUC={fold_aucs[-1]:.4f}, Acc={fold_accs[-1]:.4f}")

    print(f"CV AUC mean/std: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
    print(f"CV Acc mean/std: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")

    return roc_auc_score(y_test_np, y_proba), accuracy_score(y_test_np, y_pred)


def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [128, 256])
    weight_decay = trial.suggest_float('weight_decay', 0.0, 1e-3)
    epochs = trial.suggest_int('epochs', 10, 50)

    auc, acc = train_and_eval({'lr': lr, 'batch_size': batch_size, 'weight_decay': weight_decay, 'epochs': epochs})
    return auc


if __name__ == '__main__':
    study = op.create_study(study_name='cnn_conv', direction='maximize', storage='sqlite:///sample.db', load_if_exists=True)
    study.optimize(objective, n_trials=150)
    print('Best trial:', study.best_trial.params)
    best = study.best_trial.params
    auc, acc = train_and_eval({'lr': best['lr'], 'batch_size': best['batch_size'], 'weight_decay': best.get('weight_decay', 0.0), 'epochs': best.get('epochs', 30)})
    print(f'Final Test AUC: {auc:.4f}, Accuracy: {acc:.4f}')
