from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from src.metrics import print_core_metrics
import optuna as op
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test, X
import numpy as np

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_samples = trial.suggest_categorical("max_samples", [0.5, 0.75, 1.0])
    max_features = trial.suggest_categorical("max_features", [0.5, 0.75, 1.0])
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    bootstrap_features = trial.suggest_categorical("bootstrap_features", [True, False])
    
    model = BaggingClassifier(n_estimators=n_estimators,
                              max_samples=max_samples,
                              max_features=max_features,
                              bootstrap=bootstrap,
                              bootstrap_features=bootstrap_features,
                              random_state=42)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
    
    return scores.mean()

study = op.create_study(study_name='bagging', direction='maximize', storage='sqlite:///example.db')
study.optimize(objective, n_trials=150)
params = study.best_params
model = BaggingClassifier(**params, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
probs = model.predict_proba(X_test_scaled)[:,1]
roc_auc = roc_auc_score(y_test, probs)
print_core_metrics(y_test, y_pred)
print(f"ROC AUC: {roc_auc}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")