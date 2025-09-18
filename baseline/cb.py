from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test, X
import polars as pl
import numpy as np

pl.Config.set_tbl_rows(100)

#  Convert labels to numpy
y_train_np = np.array(y_train)
y_test_np = np.array(y_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    loss_function="Logloss",
    random_state=42,
    thread_count=-1,
    verbose=False,
)

#  SMOTE oversampling on training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train_np)

#  Cross-validation (on original labels, not Polars Series)
scores = cross_val_score(model, X_train_scaled, y_train_np, cv=cv, scoring="accuracy")

#  Train on resampled data
model.fit(X_train_res, y_train_res)

# Predictions
y_pred = model.predict(X_test_scaled)

roc_auc = roc_auc_score(y_test_np, model.predict_proba(X_test_scaled)[:, 1])
results = permutation_importance(
    model, X_test_scaled, y_test_np, n_repeats=30, random_state=42, n_jobs=-1
)

perm_imp = pl.DataFrame(
    {"Feature": X.columns, "Importance": results.importances_mean}
).sort("Importance")

print(f"Accuracy: {accuracy_score(y_test_np, y_pred)}")
print(f"Cross-validated scores: {scores}")
print(f"ROC AUC: {roc_auc}")
print(f"Classification Report:\n{classification_report(y_test_np, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test_np, y_pred)}")
print(f"Feature Importances:\n{perm_imp}")

"""  
Accuracy: 0.8527508443185532
Cross-validated scores: [0.8540146  0.85379671 0.85532193 0.85382395 0.85204957]
ROC AUC: 0.8153381376556538
Classification Report:
              precision    recall  f1-score   support

         0.0       0.87      0.97      0.92     38922
         1.0       0.54      0.21      0.31      6973

    accuracy                           0.85     45895
   macro avg       0.71      0.59      0.61     45895
weighted avg       0.82      0.85      0.82     45895

Confusion Matrix:
[[37639  1283]
 [ 5475  1498]]
Feature Importances:
shape: (20, 2)
┌──────────────────────────┬────────────┐
│ Feature                  ┆ Importance │
│ ---                      ┆ ---        │
│ str                      ┆ f64        │
╞══════════════════════════╪════════════╡
│ menthlth                 ┆ -0.000639  │
│ sex                      ┆ -0.000278  │
│ education                ┆ -0.000278  │
│ diffwalk                 ┆ -0.000263  │
│ nodocbccost              ┆ -0.000222  │
│ stroke                   ┆ -0.000139  │
│ anyhealthcare            ┆ -0.000112  │
│ income                   ┆ -0.000041  │
│ physactivity             ┆ -0.000038  │
│ physhlth                 ┆ 0.000015   │
│ smoker                   ┆ 0.000105   │
│ fruit_veggie_consumption ┆ 0.000119   │
│ cholcheck                ┆ 0.000171   │
│ heartdiseaseorattack     ┆ 0.000259   │
│ hvyalcoholconsump        ┆ 0.000365   │
│ age                      ┆ 0.002047   │
│ highbp                   ┆ 0.002135   │
│ highchol                 ┆ 0.002418   │
│ genhlth                  ┆ 0.006874   │
│ bmi                      ┆ 0.009871   │
└──────────────────────────┴────────────┘
"""
