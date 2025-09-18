from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test, X
from sklearn.inspection import permutation_importance
import polars as pl
pl.Config.set_tbl_rows(100)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = GaussianNB(var_smoothing=1e-9)
scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

results = permutation_importance(model, X_test_scaled, y_test, n_repeats=30, random_state=42)

for i in results.importances_mean.argsort()[::-1]:
    print(f"Feature {X.columns[i]}: {results.importances_mean[i]:.3f} ± {results.importances_std[i]:.3f}")

roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"ROC AUC: {roc_auc}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Cross-Validation Scores:\n{scores}")

"""Feature stroke: 0.013 ± 0.001
Feature heartdiseaseorattack: 0.004 ± 0.001
Feature bmi: 0.002 ± 0.001
Feature physhlth: 0.002 ± 0.001
Feature hvyalcoholconsump: 0.001 ± 0.000
Feature menthlth: 0.001 ± 0.001
Feature sex: 0.000 ± 0.000
Feature nodocbccost: 0.000 ± 0.000
Feature highchol: -0.000 ± 0.001
Feature diffwalk: -0.000 ± 0.001
Feature smoker: -0.000 ± 0.000
Feature anyhealthcare: -0.000 ± 0.000
Feature cholcheck: -0.001 ± 0.000
Feature education: -0.001 ± 0.000
Feature highbp: -0.001 ± 0.001
Feature fruit_veggie_consumption: -0.001 ± 0.000
Feature income: -0.001 ± 0.000
Feature genhlth: -0.001 ± 0.001
Feature physactivity: -0.001 ± 0.000
Feature age: -0.003 ± 0.001
Accuracy: 0.7582307440897701
ROC AUC: 0.771213141532728
Classification Report:
              precision    recall  f1-score   support

         0.0       0.91      0.79      0.85     38922
         1.0       0.33      0.57      0.42      6973

    accuracy                           0.76     45895
   macro avg       0.62      0.68      0.63     45895
weighted avg       0.82      0.76      0.78     45895

Confusion Matrix:
[[30838  8084]
 [ 3012  3961]]
Cross-Validation Scores:
[0.75678178 0.75465737 0.75836148 0.75819806 0.74923056]
"""
