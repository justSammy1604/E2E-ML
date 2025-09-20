from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test, X 
import optuna as op

def objective(trial):
    var_smoothing = trial.suggest_float("var_smoothing", 1e-12, 1e-7, log=True)
    model = GaussianNB(var_smoothing=var_smoothing)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    return scores.mean()

study = op.create_study(study_name='gauss', direction='maximize', storage='sqlite:///example.db')
study.optimize(objective, n_trials=100)
params = study.best_params
model = GaussianNB(**params)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
probs = model.predict_proba(X_test_scaled)[:,1]
roc_auc = roc_auc_score(y_test, probs)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"ROC AUC: {roc_auc}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
