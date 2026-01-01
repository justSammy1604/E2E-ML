from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test, X
import numpy as np
import polars as pl
import optuna as op
from src.metrics import print_core_metrics

"""Hyperparameter optimization for CatBoost with proper label conversion.

CatBoost requires labels to be array-like (NumPy / list / pandas). The original
pipeline provided Polars Series, so we convert them once up front.
"""

# Convert Polars Series to NumPy arrays (flattened)
if isinstance(y_train, pl.Series):
    y_train_array = y_train.to_numpy().ravel()
else:
    y_train_array = np.asarray(y_train).ravel()

if isinstance(y_test, pl.Series):
    y_test_array = y_test.to_numpy().ravel()
else:
    y_test_array = np.asarray(y_test).ravel()


def objective(trial):
    iterations = trial.suggest_int("iterations", 100, 1000)
    depth = trial.suggest_int("depth", 2, 16)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-5, 10.0, log=True)
    border_count = trial.suggest_int("border_count", 1, 255)
    bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0)
    random_strength = trial.suggest_float("random_strength", 1e-5, 10.0, log=True)
    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        border_count=border_count,
        bagging_temperature=bagging_temperature,
        random_strength=random_strength,
        verbose=0,
        random_state=42,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model,
        X_train_scaled,
        y_train_array,  # use NumPy array labels
        cv=cv,
        scoring="roc_auc",
    )
    return scores.mean()

study = op.create_study(
    study_name="catboost", direction="maximize", storage="sqlite:///example.db")
study.optimize(objective, n_trials=150)
params = study.best_params
model = CatBoostClassifier(**params, random_state=42, verbose=0)
model.fit(X_train_scaled, y_train_array)
y_pred = model.predict(X_test_scaled)
probs = model.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test_array, probs)
print_core_metrics(y_test_array, y_pred)
print(f"ROC AUC: {roc_auc}")
print(f"Classification Report:\n{classification_report(y_test_array, y_pred)}")