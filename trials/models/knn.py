from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import optuna as op
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test, X 

params = {
    "n_neighbors": 11,
    "weights": "distance",
    "metric": "manhattan",
    "algorithm": "ball_tree",
}
model = KNeighborsClassifier(**params)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
probs = model.predict_proba(X_test_scaled)[:,1]
roc_auc = roc_auc_score(y_test, probs)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"ROC AUC: {roc_auc}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

"""  
Accuracy: 0.8696288285995992
ROC AUC: 0.9300373439852696
Classification Report:
              precision    recall  f1-score   support

         0.0       0.90      0.87      0.89     36902
         1.0       0.82      0.87      0.85     25980

    accuracy                           0.87     62882
   macro avg       0.86      0.87      0.87     62882
weighted avg       0.87      0.87      0.87     62882

Confusion Matrix:
[[32097  4805]
 [ 3393 22587]]
"""
