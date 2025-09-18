from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test, X
import polars as pl
pl.Config.set_tbl_rows(100)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = GradientBoostingClassifier(loss='log_loss', n_estimators=50, criterion='friedman_mse', random_state=42)
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
Accuracy: 0.8543850092602681
Cross-validated scores: [0.85344264 0.85265279 0.85344264 0.85276174 0.85087839]
ROC AUC: 0.8146207195580142
Classification Report:
              precision    recall  f1-score   support

         0.0       0.86      0.98      0.92     38922
         1.0       0.59      0.14      0.22      6973

    accuracy                           0.85     45895
   macro avg       0.73      0.56      0.57     45895
weighted avg       0.82      0.85      0.81     45895

Confusion Matrix:
[[38267   655]
 [ 6028   945]]
Feature Importances:
shape: (20, 2)
┌──────────────────────────┬────────────┐
│ Feature                  ┆ Importance │
│ ---                      ┆ ---        │
│ str                      ┆ f64        │
╞══════════════════════════╪════════════╡
│ physhlth                 ┆ -0.000238  │
│ sex                      ┆ -0.000101  │
│ nodocbccost              ┆ -0.000042  │
│ education                ┆ -0.000023  │
│ smoker                   ┆ 0.0        │
│ physactivity             ┆ 0.0        │
│ anyhealthcare            ┆ 0.0        │
│ fruit_veggie_consumption ┆ 0.0        │
│ cholcheck                ┆ 0.000043   │
│ menthlth                 ┆ 0.000044   │
│ diffwalk                 ┆ 0.000134   │
│ income                   ┆ 0.000181   │
│ stroke                   ┆ 0.000182   │
│ hvyalcoholconsump        ┆ 0.000313   │
│ heartdiseaseorattack     ┆ 0.000516   │
│ age                      ┆ 0.001515   │
│ highchol                 ┆ 0.002847   │
│ highbp                   ┆ 0.003079   │
│ genhlth                  ┆ 0.005738   │
│ bmi                      ┆ 0.00751    │
└──────────────────────────┴────────────┘
"""
