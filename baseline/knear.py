from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test, X
import polars as pl
pl.Config.set_tbl_rows(100)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', n_jobs=-1)
scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

result = permutation_importance(model, X_test_scaled, y_test, n_repeats=30, random_state=42)

for i in result.importances_mean.argsort()[::-1]:
    print(f"Feature: {X.columns[i]}, Importance: {result.importances_mean[i]}")

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Cross-Validation Scores:\n{scores}")

""" Accuracy: 0.8338381087264408
Classification Report:
              precision    recall  f1-score   support

         0.0       0.87      0.94      0.91     38922
         1.0       0.41      0.22      0.29      6973

    accuracy                           0.83     45895
   macro avg       0.64      0.58      0.60     45895
weighted avg       0.80      0.83      0.81     45895

Confusion Matrix:
[[36708  2214]
 [ 5412  1561]]
Cross-Validation Scores:
[0.83070051 0.83249809 0.83533065 0.83277045 0.82843524] """
