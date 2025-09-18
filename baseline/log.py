from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test, X
import polars as pl
pl.Config.set_tbl_rows(100)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, tol=1e-4, warm_start=False)
scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
importances = model.coef_[0]

feature_importance = pl.DataFrame({"Feature": X.columns, "Importance": importances})
sorted_feature_importance = feature_importance.sort("Importance")
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Cross-Validation Scores:\n", scores)
print(f"ROC AUC: {roc_auc}")
print("Feature Importance:\n", sorted_feature_importance)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

""" Accuracy: 0.8523150670007626
Cross-Validation Scores:
 [0.85183571 0.84976577 0.85118205 0.84867633 0.84848155]
ROC AUC: 0.8091544501336693
Feature Importance:
 shape: (20, 2)
┌──────────────────────────┬────────────┐
│ Feature                  ┆ Importance │
│ ---                      ┆ ---        │
│ str                      ┆ f64        │
╞══════════════════════════╪════════════╡
│ hvyalcoholconsump        ┆ -0.797902  │
│ income                   ┆ -0.180683  │
│ education                ┆ -0.050617  │
│ physactivity             ┆ -0.040999  │
│ smoker                   ┆ -0.033587  │
│ physhlth                 ┆ -0.029631  │
│ menthlth                 ┆ -0.008046  │
│ nodocbccost              ┆ 0.003331   │
│ fruit_veggie_consumption ┆ 0.018653   │
│ anyhealthcare            ┆ 0.105128   │
│ stroke                   ┆ 0.105997   │
│ diffwalk                 ┆ 0.12137    │
│ heartdiseaseorattack     ┆ 0.215686   │
│ sex                      ┆ 0.266754   │
│ bmi                      ┆ 0.464588   │
│ age                      ┆ 0.472094   │
│ genhlth                  ┆ 0.515242   │
│ highchol                 ┆ 0.568395   │
│ highbp                   ┆ 0.72851    │
│ cholcheck                ┆ 1.248789   │
└──────────────────────────┴────────────┘
Classification Report:
               precision    recall  f1-score   support

         0.0       0.87      0.98      0.92     38922
         1.0       0.55      0.15      0.24      6973

    accuracy                           0.85     45895
   macro avg       0.71      0.57      0.58     45895
weighted avg       0.82      0.85      0.82     45895

Confusion Matrix:
 [[38038   884]
 [ 5894  1079]] """
