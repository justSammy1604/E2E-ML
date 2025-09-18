from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import optuna as op
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test

def objective(trial):
    C = trial.suggest_float("C", 1e-6, 10.0, log=True)
    tol = trial.suggest_float("tol", 1e-6, 1e-1, log=True)
    loss = trial.suggest_categorical("loss", ["hinge", "squared_hinge"])
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    model = LinearSVC(C=C, max_iter=1000, tol=tol, class_weight=class_weight, loss=loss)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    return scores.mean()

study = op.create_study(study_name='supvc', direction='maximize', storage='sqlite:///example.db')
study.optimize(objective, n_trials=100)
params = study.best_params
model = LinearSVC(**params)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)   
probs = model.decision_function(X_test_scaled)
roc_auc = roc_auc_score(y_test, probs)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"ROC AUC: {roc_auc}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")