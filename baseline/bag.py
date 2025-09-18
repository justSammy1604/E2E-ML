from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test, X
import polars as pl
pl.Config.set_tbl_rows(100)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42, n_jobs=-1)
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

""" Accuracy: 0.8340777862512256
Cross-validated scores: [0.83473145 0.83486763 0.83889857 0.83208955 0.83233011]
ROC AUC: 0.7560535526811547
Classification Report:
              precision    recall  f1-score   support

         0.0       0.87      0.95      0.91     38922
         1.0       0.41      0.21      0.28      6973

    accuracy                           0.83     45895
   macro avg       0.64      0.58      0.59     45895
weighted avg       0.80      0.83      0.81     45895

Confusion Matrix:
[[36807  2115]
 [ 5500  1473]]
Feature Importances:
shape: (20, 2)
┌──────────────────────────┬────────────┐
│ Feature                  ┆ Importance │
│ ---                      ┆ ---        │
│ str                      ┆ f64        │
╞══════════════════════════╪════════════╡
│ income                   ┆ -0.002323  │
│ sex                      ┆ -0.001804  │
│ menthlth                 ┆ -0.001458  │
│ smoker                   ┆ -0.001244  │
│ physhlth                 ┆ -0.001059  │
│ education                ┆ -0.000893  │
│ heartdiseaseorattack     ┆ -0.000782  │
│ diffwalk                 ┆ -0.000568  │
│ physactivity             ┆ -0.00055   │
│ age                      ┆ -0.000408  │
│ fruit_veggie_consumption ┆ -0.000339  │
│ highbp                   ┆ -0.000307  │
│ cholcheck                ┆ 0.000139   │
│ nodocbccost              ┆ 0.000193   │
│ highchol                 ┆ 0.000258   │
│ anyhealthcare            ┆ 0.000349   │
│ hvyalcoholconsump        ┆ 0.000406   │
│ stroke                   ┆ 0.000634   │
│ bmi                      ┆ 0.004683   │
│ genhlth                  ┆ 0.004948   │
└──────────────────────────┴────────────┘
"""
