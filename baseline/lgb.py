from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import polars as pl
import pandas as pd

pl.Config.set_tbl_rows(100)

from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test, X

if not isinstance(X_train_scaled, pd.DataFrame):
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

if not isinstance(y_train, pd.Series):
    y_train = pd.Series(y_train, name="target")
    y_test = pd.Series(y_test, name="target")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)


scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

results = permutation_importance(
    model, X_test_scaled, y_test, n_repeats=30, random_state=42, n_jobs=-1
)

perm_imp = pl.DataFrame(
    {"Feature": X.columns, "Importance": results.importances_mean}
).sort("Importance")

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Cross-validated scores: {scores}")
print(f"ROC AUC: {roc_auc}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Feature Importances:\n{perm_imp}")

"""  
Accuracy: 0.8535352434905763
Cross-validated scores: [0.90965231 0.90928243 0.91093886 0.90777074 0.90915377]
ROC AUC: 0.8158310410051092
Classification Report:
              precision    recall  f1-score   support

         0.0       0.87      0.97      0.92     38922
         1.0       0.55      0.22      0.31      6973

    accuracy                           0.85     45895
   macro avg       0.71      0.59      0.61     45895
weighted avg       0.82      0.85      0.83     45895

Confusion Matrix:
[[37666  1256]
 [ 5466  1507]]
Feature Importances:
shape: (20, 2)
┌──────────────────────────┬────────────┐
│ Feature                  ┆ Importance │
│ ---                      ┆ ---        │
│ str                      ┆ f64        │
╞══════════════════════════╪════════════╡
│ nodocbccost              ┆ -0.000143  │
│ education                ┆ -0.000086  │
│ stroke                   ┆ -0.000052  │
│ anyhealthcare            ┆ 0.00002    │
│ physactivity             ┆ 0.00015    │
│ smoker                   ┆ 0.00015    │
│ sex                      ┆ 0.000232   │
│ income                   ┆ 0.000256   │
│ cholcheck                ┆ 0.000256   │
│ heartdiseaseorattack     ┆ 0.000287   │
│ diffwalk                 ┆ 0.00029    │
│ fruit_veggie_consumption ┆ 0.000316   │
│ menthlth                 ┆ 0.000388   │
│ hvyalcoholconsump        ┆ 0.000481   │
│ physhlth                 ┆ 0.000525   │
│ age                      ┆ 0.002516   │
│ highchol                 ┆ 0.003144   │
│ highbp                   ┆ 0.003231   │
│ genhlth                  ┆ 0.007206   │
│ bmi                      ┆ 0.010199   │
└──────────────────────────┴────────────┘
"""
