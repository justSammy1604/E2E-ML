from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from src.metrics import print_core_metrics
import numpy as np
import warnings
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

warnings.filterwarnings('ignore')

# Convert Polars Series to NumPy arrays if needed
y_train_array = y_train.to_numpy() if hasattr(y_train, 'to_numpy') else np.asarray(y_train)
y_test_array = y_test.to_numpy() if hasattr(y_test, 'to_numpy') else np.asarray(y_test)

## Fixed base estimators (no grid search)
base_estimators = [
    ('rf', RandomForestClassifier(random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('ab', AdaBoostClassifier(random_state=42)),
]

stack = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=300, random_state=42),
    cv=5,
    stack_method='predict_proba',
    n_jobs=-1,
    passthrough=False,
)

# 5-fold cross-validation only (no grid search)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(stack, X_train_scaled, y_train_array, cv=cv, scoring='roc_auc', n_jobs=1)
print(f"CV ROC-AUC scores: {cv_scores}")
print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Fit on full training data
model = stack
model.fit(X_train_scaled, y_train_array)
y_pred = model.predict(X_test_scaled)
probs = model.predict_proba(X_test_scaled)[:,1]
roc_auc = roc_auc_score(y_test_array, probs)
print_core_metrics(y_test_array, y_pred)
print(f"ROC AUC: {roc_auc}")
print(f"Classification Report:\n{classification_report(y_test_array, y_pred)}")