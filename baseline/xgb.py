from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test, X
import polars as pl

pl.Config.set_tbl_rows(100)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = XGBClassifier(
	objective="binary:logistic",
	n_estimators=200,
	learning_rate=0.1,
	max_depth=6,
	subsample=0.9,
	colsample_bytree=0.9,
	random_state=42,
	n_jobs=-1,
	eval_metric="logloss",
)

# SMOTE oversampling on training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Cross-validation on original training data; fit on SMOTE-resampled data
scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
model.fit(X_train_res, y_train_res)

y_pred = model.predict(X_test_scaled)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
results = permutation_importance(model, X_test_scaled, y_test, n_repeats=30, random_state=42, n_jobs=-1)
perm_imp = pl.DataFrame({"Feature": X.columns, "Importance": results.importances_mean}).sort("Importance")

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Cross-validated scores: {scores}")
print(f"ROC AUC: {roc_auc}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Feature Importances:\n{perm_imp}")

"""  
Accuracy: 0.8537313432835821
Cross-validated scores: [0.85338817 0.85268003 0.85390565 0.85398736 0.85104181]
ROC AUC: 0.8150506593686516
Classification Report:
              precision    recall  f1-score   support

         0.0       0.87      0.97      0.92     38922
         1.0       0.55      0.22      0.31      6973

    accuracy                           0.85     45895
   macro avg       0.71      0.59      0.61     45895
weighted avg       0.82      0.85      0.83     45895

Confusion Matrix:
[[37674  1248]
 [ 5465  1508]]
Feature Importances:
shape: (20, 2)
┌──────────────────────────┬────────────┐
│ Feature                  ┆ Importance │
│ ---                      ┆ ---        │
│ str                      ┆ f64        │
╞══════════════════════════╪════════════╡
│ physactivity             ┆ 0.000004   │
│ nodocbccost              ┆ 0.000045   │
│ anyhealthcare            ┆ 0.000055   │
│ fruit_veggie_consumption ┆ 0.00009    │
│ education                ┆ 0.000129   │
│ cholcheck                ┆ 0.00021    │
│ stroke                   ┆ 0.000288   │
│ income                   ┆ 0.00033    │
│ smoker                   ┆ 0.000354   │
│ hvyalcoholconsump        ┆ 0.000365   │
│ physhlth                 ┆ 0.000372   │
│ menthlth                 ┆ 0.000384   │
│ sex                      ┆ 0.000465   │
│ heartdiseaseorattack     ┆ 0.000528   │
│ diffwalk                 ┆ 0.000666   │
│ age                      ┆ 0.002293   │
│ highbp                   ┆ 0.003345   │
│ highchol                 ┆ 0.003644   │
│ genhlth                  ┆ 0.008018   │
│ bmi                      ┆ 0.00944    │
└──────────────────────────┴────────────┘
"""
