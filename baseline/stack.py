from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test, X
import polars as pl
pl.Config.set_tbl_rows(100)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = StackingClassifier(estimators=[
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('svc', LinearSVC(random_state=42))
], final_estimator=LogisticRegression(), stack_method="auto", n_jobs=-1, passthrough=True)
scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
model.fit(X_train_scaled, y_train)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
y_pred = model.predict(X_test_scaled)

results = permutation_importance(model, X_test_scaled, y_test, n_repeats=30, random_state=42, n_jobs=-1)
perm_imp = pl.DataFrame({"Feature": X.columns, "Importance": results.importances_mean}).sort("Importance")

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Cross-validated scores: {scores}")
print(f"ROC AUC: {roc_auc}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Feature Importances:\n{perm_imp}")

""" 
Accuracy: 0.8524240113302103
Cross-validated scores: [0.85153612 0.84998366 0.85110034 0.8490304  0.84859049]
ROC AUC: 0.8093248332242741
Classification Report:
              precision    recall  f1-score   support

         0.0       0.87      0.98      0.92     38922
         1.0       0.55      0.16      0.24      6973

    accuracy                           0.85     45895
   macro avg       0.71      0.57      0.58     45895
weighted avg       0.82      0.85      0.82     45895

Confusion Matrix:
[[38041   881]
 [ 5892  1081]]
Feature Importances:
shape: (20, 2)
┌──────────────────────────┬────────────┐
│ Feature                  ┆ Importance │
│ ---                      ┆ ---        │
│ str                      ┆ f64        │
╞══════════════════════════╪════════════╡
│ smoker                   ┆ -0.00006   │
│ anyhealthcare            ┆ -0.000051  │
│ fruit_veggie_consumption ┆ 0.000001   │
│ education                ┆ 0.000029   │
│ nodocbccost              ┆ 0.00003    │
│ physactivity             ┆ 0.000079   │
│ cholcheck                ┆ 0.000139   │
│ diffwalk                 ┆ 0.00023    │
│ income                   ┆ 0.000252   │
│ menthlth                 ┆ 0.000285   │
│ hvyalcoholconsump        ┆ 0.000314   │
│ stroke                   ┆ 0.000404   │
│ sex                      ┆ 0.000518   │
│ physhlth                 ┆ 0.000773   │
│ heartdiseaseorattack     ┆ 0.000795   │
│ age                      ┆ 0.002002   │
│ highbp                   ┆ 0.00231    │
│ highchol                 ┆ 0.00232    │
│ genhlth                  ┆ 0.006582   │
│ bmi                      ┆ 0.007912   │
└──────────────────────────┴────────────┘
"""
