import torch
import numpy as np
import optuna as op
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
from sklearn.model_selection import StratifiedKFold
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test, X
from src.metrics import print_core_metrics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).to(device)

# Create numpy labels for sklearn metrics
y_train_np = y_train.numpy() if isinstance(y_train, torch.Tensor) else np.asarray(y_train)
y_test_np = y_test.cpu().numpy()


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


def train_and_eval(hparams: dict):
    model = DNNModel(X_train_scaled.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=hparams.get('weight_decay', 0.0))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    epochs = hparams.get('epochs', 30)
    batch_size = hparams.get('batch_size', 256)

    train_ds = TensorDataset(X_train_scaled.to(device), y_train.to(device))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    patience, patience_counter = 5, 0
    best_model_state = None

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

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).squeeze()
                loss = criterion(preds, yb)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        y_pred_logits = model(X_test_scaled.to(device)).squeeze()
        y_pred_probs = torch.sigmoid(y_pred_logits).cpu().numpy()
        y_pred = (y_pred_probs >= 0.5).astype(int)

    # Print core metrics and detailed report for test set
    print("\n=== Test set metrics ===")
    print_core_metrics(y_test_np, y_pred)
    print("Classification Report:\n", classification_report(y_test_np, y_pred))

    # 5-fold cross-validation on training set
    print("\n=== 5-fold cross-validation on training set ===")
    X_cv = X_train_scaled.numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    fold_accs = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_cv, y_train_np), 1):
        X_tr = torch.tensor(X_cv[tr_idx], dtype=torch.float32).to(device)
        y_tr = torch.tensor(y_train_np[tr_idx], dtype=torch.float32).to(device)
        X_val = torch.tensor(X_cv[val_idx], dtype=torch.float32).to(device)
        y_val = torch.tensor(y_train_np[val_idx], dtype=torch.float32).to(device)

        fold_model = DNNModel(X_train_scaled.shape[1]).to(device)
        opt = optim.Adam(fold_model.parameters(), lr=hparams.get('lr', 1e-3))
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
            y_val_logits = fold_model(X_val).squeeze()
            y_val_proba = torch.sigmoid(y_val_logits).cpu().numpy()
            y_val_pred = (y_val_proba >= 0.5).astype(int)
        fold_aucs.append(roc_auc_score(y_val.cpu().numpy(), y_val_proba))
        fold_accs.append(accuracy_score(y_val.cpu().numpy(), y_val_pred))
        print(f"Fold {fold}: AUC={fold_aucs[-1]:.4f}, Acc={fold_accs[-1]:.4f}")

    print(f"CV AUC mean/std: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
    print(f"CV Acc mean/std: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")

    return roc_auc_score(y_test_np, y_pred_probs), accuracy_score(y_test_np, y_pred)


def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    weight_decay = trial.suggest_float('weight_decay', 0.0, 1e-3)
    epochs = trial.suggest_int('epochs', 10, 50)

    auc, acc = train_and_eval({'lr': lr, 'batch_size': batch_size, 'weight_decay': weight_decay, 'epochs': epochs})
    return auc


if __name__ == '__main__':
    study = op.create_study(study_name='dnn_deep', direction='maximize', storage='sqlite:///sample.db', load_if_exists=True)
    study.optimize(objective, n_trials=150)
    print('Best trial:', study.best_trial.params)
    best = study.best_trial.params
    auc, acc = train_and_eval({'lr': best['lr'], 'batch_size': best['batch_size'], 'weight_decay': best.get('weight_decay', 0.0), 'epochs': best.get('epochs', 30)})
    print(f'Final Test AUC: {auc:.4f}, Accuracy: {acc:.4f}')
