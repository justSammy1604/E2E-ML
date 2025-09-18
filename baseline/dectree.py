from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test, X 
import polars as pl
pl.Config.set_tbl_rows(100)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=42, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0)
scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

importances = model.feature_importances_
feature_importances = pl.DataFrame({"feature": X.columns, "importance": importances})
sorted_feature_importances = feature_importances.sort("importance")

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"ROC AUC: {roc_auc}")
print(f"Cross-validated scores: {scores}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Feature Importances:\n{sorted_feature_importances}")

""" Accuracy: 0.7781239786469114
ROC AUC: 0.5958540172344232
Cross-validated scores: [0.77748121 0.7757381  0.77979627 0.77546574 0.77358028]
Classification Report:
              precision    recall  f1-score   support

         0.0       0.88      0.86      0.87     38922
         1.0       0.30      0.33      0.31      6973

    accuracy                           0.78     45895
   macro avg       0.59      0.59      0.59     45895
weighted avg       0.79      0.78      0.78     45895

Confusion Matrix:
[[33399  5523]
 [ 4660  2313]]
Feature Importances:
shape: (20, 2)
┌──────────────────────────┬────────────┐
│ feature                  ┆ importance │
│ ---                      ┆ ---        │
│ str                      ┆ f64        │
╞══════════════════════════╪════════════╡
│ cholcheck                ┆ 0.003875   │
│ hvyalcoholconsump        ┆ 0.008779   │
│ anyhealthcare            ┆ 0.010229   │
│ highchol                 ┆ 0.016324   │
│ stroke                   ┆ 0.016935   │
│ nodocbccost              ┆ 0.017152   │
│ heartdiseaseorattack     ┆ 0.017725   │
│ diffwalk                 ┆ 0.024297   │
│ fruit_veggie_consumption ┆ 0.027777   │
│ sex                      ┆ 0.032855   │
│ physactivity             ┆ 0.036129   │
│ smoker                   ┆ 0.042715   │
│ genhlth                  ┆ 0.063713   │
│ highbp                   ┆ 0.066413   │
│ menthlth                 ┆ 0.070233   │
│ education                ┆ 0.084327   │
│ physhlth                 ┆ 0.092956   │
│ age                      ┆ 0.105324   │
│ income                   ┆ 0.112068   │
│ bmi                      ┆ 0.150176   │
└──────────────────────────┴────────────┘
"""
