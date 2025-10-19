from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import numpy as np
import warnings
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
## No Optuna needed; using GridSearchCV

# Convert Polars Series to NumPy arrays if needed
y_train_array = y_train.to_numpy() if hasattr(y_train, 'to_numpy') else np.asarray(y_train)
y_test_array = y_test.to_numpy() if hasattr(y_test, 'to_numpy') else np.asarray(y_test)

### Base estimators (fixed configs)
base_estimators = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42)),
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('ab', AdaBoostClassifier(random_state=42)),
    ('lr', LogisticRegression(max_iter=300, random_state=42)),
]

# Force soft voting for ROC-AUC
voter = VotingClassifier(estimators=base_estimators, voting='soft')

# 5-fold cross-validation only (no grid search)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(voter, X_train_scaled, y_train_array, cv=cv, scoring='roc_auc', n_jobs=1)
print(f"CV ROC-AUC scores: {cv_scores}")
print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Fit on full training data
model = voter
model.fit(X_train_scaled, y_train_array)
y_pred = model.predict(X_test_scaled)
probs = model.predict_proba(X_test_scaled)[:,1]
roc_auc = roc_auc_score(y_test_array, probs)
print(f"Accuracy: {accuracy_score(y_test_array, y_pred)}")
print(f"ROC AUC: {roc_auc}")
print(f"Classification Report:\n{classification_report(y_test_array, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test_array, y_pred)}")

