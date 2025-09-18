from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test, X
import polars as pl
pl.Config.set_tbl_rows(100)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = VotingClassifier(estimators=[
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('lr', LogisticRegression(random_state=42))
], voting='soft', n_jobs=-1)
scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
model.fit(X_train_scaled, y_train)
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
Accuracy: 0.778516178232923
Cross-validated scores: [0.77764462 0.77581981 0.78001416 0.77557468 0.77379818]
ROC AUC: 0.7783877536022009
Classification Report:
              precision    recall  f1-score   support

         0.0       0.88      0.86      0.87     38922
         1.0       0.30      0.33      0.31      6973

    accuracy                           0.78     45895
   macro avg       0.59      0.60      0.59     45895
weighted avg       0.79      0.78      0.78     45895

Confusion Matrix:
[[33419  5503]
 [ 4662  2311]]
Feature Importances:
shape: (20, 2)
┌──────────────────────────┬────────────┐
│ Feature                  ┆ Importance │
│ ---                      ┆ ---        │
│ str                      ┆ f64        │
╞══════════════════════════╪════════════╡
│ physactivity             ┆ -0.000805  │
│ sex                      ┆ -0.000287  │
│ hvyalcoholconsump        ┆ 0.000123   │
│ cholcheck                ┆ 0.000227   │
│ anyhealthcare            ┆ 0.00028    │
│ education                ┆ 0.000298   │
│ stroke                   ┆ 0.000412   │
│ smoker                   ┆ 0.000463   │
│ fruit_veggie_consumption ┆ 0.00055    │
│ highchol                 ┆ 0.00075    │
│ nodocbccost              ┆ 0.001046   │
│ menthlth                 ┆ 0.001153   │
│ highbp                   ┆ 0.001827   │
│ heartdiseaseorattack     ┆ 0.001858   │
│ diffwalk                 ┆ 0.002399   │
│ physhlth                 ┆ 0.003999   │
│ income                   ┆ 0.004558   │
│ bmi                      ┆ 0.005273   │
│ age                      ┆ 0.00611    │
│ genhlth                  ┆ 0.011599   │
└──────────────────────────┴────────────┘
"""
