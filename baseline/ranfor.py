from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test, X
import polars as pl
pl.Config.set_tbl_rows(100)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

importance = model.feature_importances_
feature_importance = pl.DataFrame({"feature": X.columns, "importance": importance})
sorted_feature_importance = feature_importance.sort("importance")

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Cross-validated scores: {scores}")
print(f"ROC AUC: {roc_auc}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Feature Importances:\n{sorted_feature_importance}")


""" Accuracy: 0.8433380542542761
Cross-validated scores: [0.84121364 0.84306569 0.84459091 0.84175836 0.84025603]
ROC AUC: 0.7737625578242278
Classification Report:
              precision    recall  f1-score   support

         0.0       0.87      0.96      0.91     38922
         1.0       0.46      0.19      0.26      6973

    accuracy                           0.84     45895
   macro avg       0.66      0.57      0.59     45895
weighted avg       0.81      0.84      0.81     45895

Confusion Matrix:
[[37414  1508]
 [ 5682  1291]]
Feature Importances:
shape: (20, 2)
┌──────────────────────────┬────────────┐
│ feature                  ┆ importance │
│ ---                      ┆ ---        │
│ str                      ┆ f64        │
╞══════════════════════════╪════════════╡
│ cholcheck                ┆ 0.003791   │
│ hvyalcoholconsump        ┆ 0.007889   │
│ anyhealthcare            ┆ 0.009128   │
│ stroke                   ┆ 0.013415   │
│ nodocbccost              ┆ 0.015678   │
│ heartdiseaseorattack     ┆ 0.018319   │
│ fruit_veggie_consumption ┆ 0.022884   │
│ diffwalk                 ┆ 0.023487   │
│ highchol                 ┆ 0.025711   │
│ sex                      ┆ 0.027747   │
│ physactivity             ┆ 0.028805   │
│ smoker                   ┆ 0.034019   │
│ highbp                   ┆ 0.04142    │
│ menthlth                 ┆ 0.067822   │
│ genhlth                  ┆ 0.06916    │
│ education                ┆ 0.075334   │
│ physhlth                 ┆ 0.089119   │
│ income                   ┆ 0.10619    │
│ age                      ┆ 0.127152   │
│ bmi                      ┆ 0.192931   │
└──────────────────────────┴────────────┘
"""
