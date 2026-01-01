import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score

def _to_numpy(arr):
    try:
        import torch  # local import to avoid hard dependency at import time
        if hasattr(arr, 'detach') and isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
    except Exception:
        pass
    # Polars / pandas have to_numpy; otherwise np.asarray
    if hasattr(arr, 'to_numpy'):
        return arr.to_numpy()
    return np.asarray(arr)


def compute_metrics(y_true, y_pred, y_proba=None):
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    
    if y_proba is not None:
        y_proba = _to_numpy(y_proba)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
        if 0 in np.unique(y_true):
            tn = int((y_true == 0).sum())
        if 1 in np.unique(y_true):
            tp = int((y_true == 1).sum())
            
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    auc = np.nan
    if y_proba is not None:
        try:
            # Handle cases where only one class is present in y_true
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            pass

    return {
        "Accuracy": float(accuracy),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "F1 Score": float(f1),
        "AUC": float(auc),
        "Confusion Matrix": cm,
    }


def print_core_metrics(y_true, y_pred, prefix: str | None = None):
    m = compute_metrics(y_true, y_pred)
    p = f"{prefix}: " if prefix else ""
    print(f"{p}Accuracy = {m['Accuracy']:.4f}, Sensitivity = {m['Sensitivity']:.4f}, Specificity = {m['Specificity']:.4f}")
    print(f"{p}Confusion Matrix:\n{m['Confusion Matrix']}")


""" 
=== Faster Chi2-AUC-Sparsity Hybrid for Diabetes FS ===
GWO-Chi2 AUC Sparsity - Best Fitness: 0.8683
Number of Selected Features: 15
Choice Percentage: 75.00%
Selected Feature Indices: [ 1  2  3  4  5  6  9 10 11 12 13 14 15 16 18]
Selected Feature Names: ['highchol', 'cholcheck', 'bmi', 'smoker', 'stroke', 'heartdiseaseorattack', 'anyhealthcare', 'nodocbccost', 'genhlth', 'menthlth', 'physhlth', 'diffwalk', 'sex', 'age', 'income']

WOA-Chi2 AUC Sparsity - Best Fitness: 0.8467
Number of Selected Features: 12
Choice Percentage: 60.00%
Selected Feature Indices: [ 1  3  5  9 11 12 13 14 15 16 17 19]
Selected Feature Names: ['highchol', 'bmi', 'stroke', 'anyhealthcare', 'genhlth', 'menthlth', 'physhlth', 'diffwalk', 'sex', 'age', 'education', 'fruit_veggie_consumption']

FA-Chi2 AUC Sparsity - Best Fitness: 1.0710
Number of Selected Features: 13
Choice Percentage: 65.00%
Selected Feature Indices: [ 1  2  4  6  7  8  9 10 11 13 14 15 17]
Selected Feature Names: ['highchol', 'cholcheck', 'smoker', 'heartdiseaseorattack', 'physactivity', 'hvyalcoholconsump', 'anyhealthcare', 'nodocbccost', 'genhlth', 'physhlth', 'diffwalk', 'sex', 'education']       


--- Performance for GWO_Chi2 selected features ---
XGBoost: Acc = 0.8857, Sensitivity = 0.7469, Specificity = 0.9827, F1 = 0.8431, AUC = 0.9460
GradientBoost: Acc = 0.8898, Sensitivity = 0.7682, Specificity = 0.9747, F1 = 0.8514, AUC = 0.9463
CatBoost: Acc = 0.8909, Sensitivity = 0.7641, Specificity = 0.9794, F1 = 0.8520, AUC = 0.9463
LGBMClassifier: Acc = 0.8901, Sensitivity = 0.7660, Specificity = 0.9768, F1 = 0.8514, AUC = 0.9469
BaggingClassifier: Acc = 0.8870, Sensitivity = 0.7298, Specificity = 0.9967, F1 = 0.8415, AUC = 0.9466

--- Performance for WOA_Chi2 selected features ---
XGBoost: Acc = 0.8817, Sensitivity = 0.7363, Specificity = 0.9832, F1 = 0.8365, AUC = 0.9419
GradientBoost: Acc = 0.8856, Sensitivity = 0.7578, Specificity = 0.9748, F1 = 0.8449, AUC = 0.9434
CatBoost: Acc = 0.8867, Sensitivity = 0.7551, Specificity = 0.9786, F1 = 0.8457, AUC = 0.9433
LGBMClassifier: Acc = 0.8861, Sensitivity = 0.7574, Specificity = 0.9759, F1 = 0.8453, AUC = 0.9439
BaggingClassifier: Acc = 0.8841, Sensitivity = 0.7233, Specificity = 0.9965, F1 = 0.8369, AUC = 0.9438

--- Performance for FA_Chi2 selected features ---
XGBoost: Acc = 0.8735, Sensitivity = 0.7004, Specificity = 0.9943, F1 = 0.8199, AUC = 0.9248
GradientBoost: Acc = 0.8750, Sensitivity = 0.7123, Specificity = 0.9885, F1 = 0.8241, AUC = 0.9259
CatBoost: Acc = 0.8754, Sensitivity = 0.7080, Specificity = 0.9924, F1 = 0.8237, AUC = 0.9265
LGBMClassifier: Acc = 0.8752, Sensitivity = 0.7099, Specificity = 0.9906, F1 = 0.8239, AUC = 0.9268
BaggingClassifier: Acc = 0.8745, Sensitivity = 0.6973, Specificity = 0.9982, F1 = 0.8204, AUC = 0.9255

--- Performance for All Features selected features ---
XGBoost: Acc = 0.8939, Sensitivity = 0.7687, Specificity = 0.9813, F1 = 0.8562, AUC = 0.9518
GradientBoost: Acc = 0.8951, Sensitivity = 0.7834, Specificity = 0.9730, F1 = 0.8600, AUC = 0.9510
CatBoost: Acc = 0.8966, Sensitivity = 0.7801, Specificity = 0.9780, F1 = 0.8612, AUC = 0.9511
LGBMClassifier: Acc = 0.8962, Sensitivity = 0.7828, Specificity = 0.9754, F1 = 0.8611, AUC = 0.9517
BaggingClassifier: Acc = 0.8940, Sensitivity = 0.7487, Specificity = 0.9955, F1 = 0.8531, AUC = 0.9523
"""
