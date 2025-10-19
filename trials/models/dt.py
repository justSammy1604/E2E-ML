from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test, X
import optuna as op

def objective(trial):
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    splitter = trial.suggest_categorical("splitter", ["best", "random"])
    max_depth = trial.suggest_int("max_depth", 1, 100)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 100)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 100)
    min_weight_fraction_leaf = trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.5)
    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 2, 200)
    min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0.0, 0.5)
    ccp_alpha = trial.suggest_float("ccp_alpha", 0.0, 0.5)
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    
    model = DecisionTreeClassifier(max_depth=max_depth, 
                                   criterion=criterion,
                                   min_samples_split=min_samples_split, 
                                   min_samples_leaf=min_samples_leaf, 
                                   class_weight=class_weight,
                                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                                    max_leaf_nodes=max_leaf_nodes,
                                    min_impurity_decrease=min_impurity_decrease,
                                    splitter=splitter,
                                    ccp_alpha=ccp_alpha,
                                   random_state=42)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    
    return scores.mean()

study = op.create_study(study_name='decision_tree', direction='maximize', storage='sqlite:///example.db')
study.optimize(objective, n_trials=150)
params = study.best_params
model = DecisionTreeClassifier(**params, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
probs = model.predict_proba(X_test_scaled)[:,1]
roc_auc = roc_auc_score(y_test, probs)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"ROC AUC: {roc_auc}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

""" class DecisionTreeClassifier(
    criterion: Literal['gini', 'entropy', 'log_loss'] = "gini",
    splitter: Literal['best', 'random'] = "best",
    max_depth: Int | None = None,
    min_samples_split: float = 2,
    min_samples_leaf: float = 1,
    min_weight_fraction_leaf: Float = 0,
    max_features: float | Literal['auto', 'sqrt', 'log2'] | None = None,
    random_state: Int | None = None,
    max_leaf_nodes: Int | None = None,
    min_impurity_decrease: Float = 0,
    class_weight: Mapping | str | Sequence[Mapping] | None = None,
    ccp_alpha: float = 0
)
 """
