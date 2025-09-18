from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import  LinearSVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import polars as pl
pl.Config.set_tbl_rows(100)
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test, X

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = LinearSVC(penalty="l2", loss="squared_hinge", dual=True, tol=1e-4, C=1.0, multi_class="ovr", fit_intercept=True, intercept_scaling=1, class_weight=None)
scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy") 
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
roc_auc = roc_auc_score(y_test, model.decision_function(X_test_scaled))

importance = permutation_importance(model, X_test_scaled, y_test, n_repeats=30, random_state=42, n_jobs=-1)
feature_importance = pl.DataFrame({"feature": X.columns, "importance": importance.importances_mean})
sorted_feature_importance = feature_importance.sort("importance")

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Cross-validated scores: {scores}")
print(f"ROC AUC: {roc_auc}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Feature importances:\n{sorted_feature_importance}")

""" LINEAR SVC METRICS
Accuracy: 0.8514653012310709
Cross-validated scores: [0.84973853 0.84878527 0.85115481 0.8490304  0.84902628]
ROC AUC: 0.8089329198760165
Classification Report:
              precision    recall  f1-score   support

         0.0       0.86      0.99      0.92     38922
         1.0       0.60      0.06      0.12      6973

    accuracy                           0.85     45895
   macro avg       0.73      0.53      0.52     45895
weighted avg       0.82      0.85      0.80     45895

Confusion Matrix:
[[38626   296]
 [ 6521   452]]
Feature importances:
shape: (20, 2)
┌──────────────────────────┬────────────┐
│ feature                  ┆ importance │
│ ---                      ┆ ---        │
│ str                      ┆ f64        │
╞══════════════════════════╪════════════╡
│ physhlth                 ┆ -0.000235  │
│ menthlth                 ┆ -0.000214  │
│ sex                      ┆ -0.000182  │
│ physactivity             ┆ -0.000026  │
│ fruit_veggie_consumption ┆ 0.000009   │
│ nodocbccost              ┆ 0.000018   │
│ education                ┆ 0.000022   │
│ smoker                   ┆ 0.000026   │
│ anyhealthcare            ┆ 0.000062   │
│ stroke                   ┆ 0.000127   │
│ cholcheck                ┆ 0.000137   │
│ hvyalcoholconsump        ┆ 0.000277   │
│ income                   ┆ 0.000492   │
│ diffwalk                 ┆ 0.000743   │
│ heartdiseaseorattack     ┆ 0.000909   │
│ age                      ┆ 0.001047   │
│ highbp                   ┆ 0.001507   │
│ highchol                 ┆ 0.001513   │
│ genhlth                  ┆ 0.003422   │
│ bmi                      ┆ 0.004358   │
└──────────────────────────┴────────────┘ """
