from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from src.metrics import print_core_metrics
import optuna as op
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test
import numpy as np

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    num_leaves = trial.suggest_int("num_leaves", 20, 150)
    max_depth = trial.suggest_int("max_depth", 2, 20)
    min_child_samples = trial.suggest_int("min_child_samples", 5, 100)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    reg_alpha = trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True)
    reg_lambda = trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True)
    
    model = LGBMClassifier(n_estimators=n_estimators,
                           learning_rate=learning_rate,
                           num_leaves=num_leaves,
                           max_depth=max_depth,
                           min_child_samples=min_child_samples,
                           subsample=subsample,
                           colsample_bytree=colsample_bytree,
                           reg_alpha=reg_alpha,
                           reg_lambda=reg_lambda,
                           random_state=42)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
    
    return scores.mean()

study = op.create_study(study_name='lgbm', direction='maximize', storage='sqlite:///example.db')
study.optimize(objective, n_trials=150)
params = study.best_params
model = LGBMClassifier(**params, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
probs = model.predict_proba(X_test_scaled)[:,1]
roc_auc = roc_auc_score(y_test, probs)
print_core_metrics(y_test, y_pred)
print(f"ROC AUC: {roc_auc}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")