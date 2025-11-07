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
import optuna as op
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Convert labels to numpy
y_train_np = y_train.to_numpy() if hasattr(y_train, 'to_numpy') else np.asarray(y_train)
y_test_np = y_test.to_numpy() if hasattr(y_test, 'to_numpy') else np.asarray(y_test)

# GPU acceleration setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Reshape for RNN: (samples, timesteps=features, input_dim=1)
X_train = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(-1).to(device)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(-1).to(device)
y_train = torch.tensor(y_train_np, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test_np, dtype=torch.float32).to(device)

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

class_weights = compute_class_weight(
    "balanced", classes=np.unique(y_train.cpu()), y=y_train.cpu().numpy()
)
weights = torch.tensor(class_weights, dtype=torch.float32).to(device)


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


def train_and_eval(hparams: dict):
    # Build model with hyperparameters
    model = RNN_LSTM(X_train.shape[1]).to(device)
    if hparams.get('weight_decay', 0.0) > 0:
        optimizer = optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=hparams['lr'])

    criterion = nn.BCELoss()

    EPOCHS = hparams.get('epochs', 30)
    batch_size = hparams.get('batch_size', 256)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_proba = model(X_test).squeeze().cpu().numpy()
        y_pred = (y_proba >= 0.5).astype(int)

    # Print classification report and confusion matrix
    print("\n=== Test set classification report ===")
    print(classification_report(y_test_np, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test_np, y_pred))

    # 5-fold cross-validation on training set
    print("\n=== 5-fold cross-validation on training set ===")
    # X_train is (n_samples, timesteps, 1) -> flatten timesteps for sklearn split
    X_cv = X_train.cpu().numpy()
    X_cv_flat = X_cv.reshape(X_cv.shape[0], -1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    fold_accs = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_cv_flat, y_train_np), 1):
        X_tr = torch.tensor(X_cv_flat[tr_idx], dtype=torch.float32).to(device).reshape(-1, X_train.shape[1], 1)
        y_tr = torch.tensor(y_train_np[tr_idx], dtype=torch.float32).to(device)
        X_val = torch.tensor(X_cv_flat[val_idx], dtype=torch.float32).to(device).reshape(-1, X_train.shape[1], 1)
        y_val = torch.tensor(y_train_np[val_idx], dtype=torch.float32).to(device)

        fold_model = RNN_LSTM(X_train.shape[1]).to(device)
        opt = optim.Adam(fold_model.parameters(), lr=hparams.get('lr', 1e-3))
        crit = nn.BCELoss()
        cv_epochs = min(10, EPOCHS)
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
    # Compact hyperparameter space
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [128, 256])
    weight_decay = trial.suggest_float('weight_decay', 0.0, 1e-3)
    epochs = trial.suggest_int('epochs', 10, 50)

    auc, acc = train_and_eval({'lr': lr, 'batch_size': batch_size, 'weight_decay': weight_decay, 'epochs': epochs})
    return auc


if __name__ == '__main__':
    study = op.create_study(study_name='rnn_rec', direction='maximize', storage='sqlite:///sample.db', load_if_exists=True)
    study.optimize(objective, n_trials=150)
    print('Best trial:', study.best_trial.params)
    # final training with best params
    best = study.best_trial.params
    auc, acc = train_and_eval({'lr': best['lr'], 'batch_size': best['batch_size'], 'weight_decay': best.get('weight_decay', 0.0), 'epochs': best.get('epochs', 30)})
    print(f'Final Test AUC: {auc:.4f}, Accuracy: {acc:.4f}')
