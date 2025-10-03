from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test, X
import polars as pl 
pl.Config.set_tbl_rows(100)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = AdaBoostClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

features = model.feature_importances_
feature_importances = pl.DataFrame({"Feature": X.columns, "Importance": features})
sorted_feature_importances = feature_importances.sort("Importance")

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Cross-validated scores: {scores}")
print(f"ROC AUC: {roc_auc}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Feature Importances:\n{sorted_feature_importances}")

""" Accuracy: 0.8524240113302103
Cross-validated scores: [0.8516723  0.8513727  0.85208084 0.85031049 0.84965273]
ROC AUC: 0.8102469376308465
Classification Report:
              precision    recall  f1-score   support

         0.0       0.87      0.97      0.92     38922
         1.0       0.54      0.18      0.27      6973

    accuracy                           0.85     45895
   macro avg       0.71      0.58      0.59     45895
weighted avg       0.82      0.85      0.82     45895

Confusion Matrix:
[[37865  1057]
 [ 5716  1257]]
Feature Importances:
shape: (20, 2)
┌──────────────────────────┬────────────┐
│ Feature                  ┆ Importance │
│ ---                      ┆ ---        │
│ str                      ┆ f64        │
╞══════════════════════════╪════════════╡
│ smoker                   ┆ 0.0        │
│ stroke                   ┆ 0.0        │
│ physactivity             ┆ 0.0        │
│ anyhealthcare            ┆ 0.0        │
│ nodocbccost              ┆ 0.0        │
│ physhlth                 ┆ 0.0        │
│ diffwalk                 ┆ 0.0        │
│ education                ┆ 0.0        │
│ fruit_veggie_consumption ┆ 0.0        │
│ menthlth                 ┆ 0.007089   │
│ heartdiseaseorattack     ┆ 0.01043    │
│ sex                      ┆ 0.013356   │
│ income                   ┆ 0.015394   │
│ highchol                 ┆ 0.029973   │
│ hvyalcoholconsump        ┆ 0.052345   │
│ bmi                      ┆ 0.117808   │
│ cholcheck                ┆ 0.123296   │
│ genhlth                  ┆ 0.149906   │
│ age                      ┆ 0.154595   │
│ highbp                   ┆ 0.325809   │
└──────────────────────────┴────────────┘
"""
