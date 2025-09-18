from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import optuna as op
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test, X

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    max_leaf_nodes = trial.suggest_categorical("max_leaf_nodes", [None, 10, 20, 30, 40, 50])
    min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0.0, 0.1)
    max_samples = trial.suggest_categorical("max_samples", [None, 0.5, 0.75, 1.0])
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   class_weight=class_weight,
                                   max_features=max_features,
                                   max_leaf_nodes=max_leaf_nodes,
                                   min_impurity_decrease=min_impurity_decrease,
                                   max_samples=max_samples,
                                   random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    return scores.mean()

study = op.create_study(study_name='random_forest', direction='maximize', storage='sqlite:///example.db')
study.optimize(objective, n_trials=100) 
params = study.best_params
model = RandomForestClassifier(**params, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
probs = model.predict_proba(X_test_scaled)[:,1]
roc_auc = roc_auc_score(y_test, probs)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"ROC AUC: {roc_auc}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

""" 
min_impurity_decrease
min_samples_leaf
n_estimators
max_depth
max_leaf_nodes
min_samples_split
"""