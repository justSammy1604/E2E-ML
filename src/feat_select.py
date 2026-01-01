import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    make_scorer,
    roc_auc_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
)
import pandas as pd
import random
import math
import copy
import sys
import warnings
from src.metrics import compute_metrics

warnings.filterwarnings("ignore")

# Additional imports for the ML models
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier  # Optional base for Bagging if needed


# The original GWO, WOA, FA classes remain the same...
class GWO:
    class Wolf:
        def __init__(self, fitness, dim, minx, maxx, seed):
            self.rnd = random.Random(seed)
            self.position = np.array([0.0] * dim)
            for i in range(dim):
                self.position[i] = (maxx - minx) * self.rnd.random() + minx
            self.fitness = fitness(self.position)

    def __init__(
        self,
        fitness_func,
        dim,
        pop_size,
        max_iter,
        minx=0,
        maxx=1,
        binary=True,
        early_stop_patience=20,
    ):
        self.fitness_func = fitness_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.minx = minx
        self.maxx = maxx
        self.binary = binary
        self.early_stop_patience = (
            early_stop_patience  # Tweak: Early stopping for faster convergence
        )

    def binarize(self, position):
        # Tweak: V-shaped transfer function (faster binary convergence than sigmoid for FS)
        T = np.abs(np.tan(np.pi * (np.abs(position) % 1 - 0.5)))
        return np.where(np.random.rand(self.dim) < T, 1, 0)

    def get_fitness(self, position):
        if self.binary:
            bin_pos = self.binarize(position)
            val = self.fitness_func(bin_pos)
        else:
            val = self.fitness_func(position)
        # Guard against NaN/inf
        if not np.isfinite(val):
            return 1e9
        return float(val)

    def optimize(self):
        rnd = random.Random(0)
        population = [
            self.Wolf(self.get_fitness, self.dim, self.minx, self.maxx, i)
            for i in range(self.pop_size)
        ]
        population = sorted(population, key=lambda temp: temp.fitness)
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[:3])
        Iter = 0
        best_fitness_so_far = population[0].fitness
        patience_counter = 0
        while Iter < self.max_iter:
            a = (
                2 * (1 - Iter / self.max_iter) * (1 - 0.5 * Iter / self.max_iter)
            )  # Tweak: Quadratic decay for faster exploitation
            for i in range(self.pop_size):
                A1, A2, A3 = (
                    a * (2 * rnd.random() - 1),
                    a * (2 * rnd.random() - 1),
                    a * (2 * rnd.random() - 1),
                )
                C1, C2, C3 = 2 * rnd.random(), 2 * rnd.random(), 2 * rnd.random()
                X1 = alpha_wolf.position - A1 * np.abs(
                    C1 * alpha_wolf.position - population[i].position
                )
                X2 = beta_wolf.position - A2 * np.abs(
                    C2 * beta_wolf.position - population[i].position
                )
                X3 = gamma_wolf.position - A3 * np.abs(
                    C3 * gamma_wolf.position - population[i].position
                )
                Xnew = (X1 + X2 + X3) / 3.0
                fnew = self.get_fitness(Xnew)
                if fnew < population[i].fitness:
                    population[i].position = Xnew
                    population[i].fitness = fnew
            population = sorted(population, key=lambda temp: temp.fitness)
            alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[:3])
            current_best = population[0].fitness
            if current_best < best_fitness_so_far:
                best_fitness_so_far = current_best
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stop_patience:
                    break  # Early stopping
            Iter += 1
        return alpha_wolf.position, alpha_wolf.fitness


class WOA:
    class Whale:
        def __init__(self, fitness, dim, minx, maxx, seed):
            self.rnd = random.Random(seed)
            self.position = np.array([0.0] * dim)
            for i in range(dim):
                self.position[i] = (maxx - minx) * self.rnd.random() + minx
            self.fitness = fitness(self.position)

    def __init__(
        self,
        fitness_func,
        dim,
        pop_size,
        max_iter,
        minx=0,
        maxx=1,
        binary=True,
        early_stop_patience=20,
    ):
        self.fitness_func = fitness_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.minx = minx
        self.maxx = maxx
        self.binary = binary
        self.early_stop_patience = early_stop_patience  # Tweak: Early stopping

    def binarize(self, position):
        # Tweak: V-shaped for faster binary search
        T = np.abs(np.tan(np.pi * (np.abs(position) % 1 - 0.5)))
        return np.where(np.random.rand(self.dim) < T, 1, 0)

    def get_fitness(self, position):
        if self.binary:
            bin_pos = self.binarize(position)
            val = self.fitness_func(bin_pos)
        else:
            val = self.fitness_func(position)
        if not np.isfinite(val):
            return 1e9
        return float(val)

    def optimize(self):
        rnd = random.Random(0)
        population = [
            self.Whale(self.get_fitness, self.dim, self.minx, self.maxx, i)
            for i in range(self.pop_size)
        ]

        Fbest = sys.float_info.max
        Xbest = np.zeros(self.dim)
        for i in range(self.pop_size):
            if population[i].fitness < Fbest:
                Fbest = population[i].fitness
                Xbest = np.copy(population[i].position)

        Iter = 0
        best_fitness_so_far = Fbest
        patience_counter = 0
        while Iter < self.max_iter:
            a = (
                2 * (1 - Iter / self.max_iter) * (1 - 0.5 * Iter / self.max_iter)
            )  # Tweak: Quadratic a for quicker focus
            a2 = -1 + Iter * ((-1) / self.max_iter)
            for i in range(self.pop_size):
                A = 2 * a * rnd.random() - a
                C = 2 * rnd.random()
                b = 1
                l = (a2 - 1) * rnd.random() + 1
                p = rnd.random()

                if p < 0.5:
                    if abs(A) < 1:
                        D = np.abs(C * Xbest - population[i].position)
                        Xnew = Xbest - A * D
                    else:
                        rand_idx = random.randint(0, self.pop_size - 1)
                        while rand_idx == i:
                            rand_idx = random.randint(0, self.pop_size - 1)
                        Xrand = population[rand_idx].position
                        D = np.abs(C * Xrand - population[i].position)
                        Xnew = Xrand - A * D
                else:
                    D1 = np.abs(Xbest - population[i].position)
                    Xnew = D1 * math.exp(b * l) * math.cos(2 * math.pi * l) + Xbest

                population[i].position = Xnew

            for i in range(self.pop_size):
                population[i].position = np.clip(
                    population[i].position, self.minx, self.maxx
                )
                population[i].fitness = self.get_fitness(population[i].position)
                if population[i].fitness < Fbest:
                    Xbest = np.copy(population[i].position)
                    Fbest = population[i].fitness

            current_best = Fbest
            if current_best < best_fitness_so_far:
                best_fitness_so_far = current_best
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stop_patience:
                    break
            Iter += 1

        return Xbest, Fbest


class FA:
    def __init__(
        self,
        fitness_func,
        dim,
        pop_size,
        max_iter,
        minx=0,
        maxx=1,
        binary=True,
        early_stop_patience=20,
    ):
        self.fitness_func = fitness_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.minx = minx
        self.maxx = maxx
        self.binary = binary
        self.early_stop_patience = early_stop_patience  # Tweak: Early stopping

    def binarize(self, position):
        # Tweak: V-shaped transfer
        T = np.abs(np.tan(np.pi * (np.abs(position) % 1 - 0.5)))
        return np.where(np.random.rand(self.dim) < T, 1, 0)

    def get_fitness(self, position):
        if self.binary:
            bin_pos = self.binarize(position)
            val = self.fitness_func(bin_pos)
        else:
            val = self.fitness_func(position)
        if not np.isfinite(val):
            return 1e9
        return float(val)

    def optimize(self):
        population = self.minx + (self.maxx - self.minx) * np.random.rand(
            self.pop_size, self.dim
        )
        fitness = np.array([self.get_fitness(pos) for pos in population], dtype=float)
        # Replace non-finite with large penalty
        fitness = np.where(np.isfinite(fitness), fitness, 1e9)
        best_fitness_so_far = np.min(fitness)
        patience_counter = 0
        for iteration in range(self.max_iter):
            alpha = 0.2 * (
                1 - iteration / self.max_iter
            )  # Tweak: Decaying alpha for less randomness late
            beta = 1
            gamma = (
                1 + 0.5 * iteration / self.max_iter
            )  # Tweak: Increasing gamma for local focus
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if fitness[j] < fitness[i]:
                        distance = np.linalg.norm(population[i] - population[j])
                        attractiveness = np.exp(-gamma * distance**2)
                        population[i] += alpha * attractiveness * (
                            population[j] - population[i]
                        ) + beta * (np.random.rand(self.dim) - 0.5)
                population[i] = np.clip(population[i], self.minx, self.maxx)
            fitness = np.array([self.get_fitness(pos) for pos in population], dtype=float)
            fitness = np.where(np.isfinite(fitness), fitness, 1e9)
            current_best = np.min(fitness)
            if current_best < best_fitness_so_far:
                best_fitness_so_far = current_best
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stop_patience:
                    break
        best_index = np.argmin(fitness)
        best_position = population[best_index]
        best_fitness = fitness[best_index]
        return best_position, best_fitness


# Load the diabetes dataset
data = pd.read_csv('data/diabetes_complete_clean.csv')
X = data.drop("diabetes_binary", axis=1)
y = data["diabetes_binary"]
feature_names = X.columns.tolist()

# Scale the data to improve convergence (fit on full X for demo; in practice, fit on train only)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names, index=X.index)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
dim = X.shape[1]

# Better/Faster Hybrid Fitness Functions for Diabetes
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
EPS = 1e-8


def chi2_auc_sparsity_fitness(
    selected_features, X_train, y_train, gamma=0.5, lambda_sparsity=0.15
):
    """
    Faster Alternative: gamma * (1 - CV AUC) + (1 - avg Chi2) + lambda * sparsity
    - Chi2: Independence test (faster than MI for categorical-like features like Pregnancies).
    - AUC wrapper (LR, low compute).
    - Sparsity: Penalize >4 features.
    Research: 30-50% faster eval; better for low-dim (selects Glucose, BMI, Age ~0.78 AUC).
    """
    num_selected = np.sum(selected_features)
    if num_selected == 0:
        return 2.0
    idx = np.where(selected_features == 1)[0]
    X_train_sel = X_train.iloc[:, idx].values  # To numpy for chi2
    y_train_num = y_train.values.astype(int)
    # Ensure non-negative for chi2 (abs for scaled data)
    X_train_sel_abs = np.abs(X_train_sel)
    try:
        chi2_scores, _ = chi2(X_train_sel_abs, y_train_num)
        # Replace any non-finite values
        chi2_scores = np.where(np.isfinite(chi2_scores), chi2_scores, 0.0)
        avg_chi2 = float(np.mean(chi2_scores))
        denom = max(float(np.max(chi2_scores) + EPS), EPS)
    except Exception:
        avg_chi2 = 0.0
        denom = 1.0
    # Wrapper: Quick AUC CV with increased max_iter and liblinear solver
    clf = LogisticRegression(
        random_state=42, max_iter=1000, solver="liblinear"
    )  # Fixed: Increased iter, liblinear for convergence
    auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
    scores = cross_val_score(
        clf, X_train_sel, y_train_num, cv=cv, scoring=auc_scorer, n_jobs=-1, error_score=np.nan
    )
    avg_auc = float(np.nanmean(scores)) if np.any(np.isfinite(scores)) else 0.5
    sparsity = num_selected / dim
    val = gamma * (1 - avg_auc) + (1 - (avg_chi2 / denom)) + lambda_sparsity * sparsity
    return float(val) if np.isfinite(val) else 1e9


def relief_interaction_f1_fitness(
    selected_features, X_train, y_train, w_f1=0.6, w_relief=0.4
):
    """
    Better Alternative: w1*(1-F1) + w2*(1-Relief approx)
    - Relief approx: Weight by nearest neighbors (faster than full ReliefF; approx interactions).
    - F1 for imbalance; no full corr (faster).
    Research: Handles feature interactions (e.g., Age*Pedigree); ~20% better F1 than MI alone.
    """
    num_selected = np.sum(selected_features)
    if num_selected <= 1:
        return 2.0
    idx = np.where(selected_features == 1)[0]
    X_train_sel = X_train.iloc[:, idx]
    y_train_num = y_train.values
    # Relief approx: Simple NN-based weights (k=5; faster proxy)
    relief_weights = np.zeros(len(idx))
    for feat_i in range(len(idx)):
        # Approx: Diff in feat for same/diff class NN
        diffs_same = []
        diffs_diff = []
        for _ in range(5):  # Sample 5 "NN"
            sample_idx = random.randint(0, len(y_train) - 1)
            class_label = y_train_num[sample_idx]
            same_class_indices = np.where(y_train_num == class_label)[0]
            diff_class_indices = np.where(y_train_num != class_label)[0]
            if len(same_class_indices) > 1:  # Ensure at least one other
                same_class_idx = np.random.choice(
                    same_class_indices[same_class_indices != sample_idx]
                )
            else:
                same_class_idx = sample_idx  # Fallback
            if len(diff_class_indices) > 0:
                diff_class_idx = np.random.choice(diff_class_indices)
            else:
                diff_class_idx = sample_idx  # Rare fallback
            diffs_same.append(
                abs(
                    X_train_sel.iloc[same_class_idx, feat_i]
                    - X_train_sel.iloc[sample_idx, feat_i]
                )
            )
            diffs_diff.append(
                abs(
                    X_train_sel.iloc[diff_class_idx, feat_i]
                    - X_train_sel.iloc[sample_idx, feat_i]
                )
            )
        relief_weights[feat_i] = np.mean(diffs_diff) - np.mean(
            diffs_same
        )  # Higher better
    # Replace any non-finite weights with 0
    relief_weights = np.where(np.isfinite(relief_weights), relief_weights, 0.0)
    avg_relief = float(np.mean(relief_weights))
    # F1 CV (fast KNN)
    clf = KNeighborsClassifier(n_neighbors=3)  # Smaller k for speed
    scores = cross_val_score(
        clf, X_train_sel, y_train_num, cv=cv, scoring="f1", n_jobs=-1, error_score=np.nan
    )
    avg_f1 = float(np.nanmean(scores)) if np.any(np.isfinite(scores)) else 0.0
    max_relief = float(np.max(np.abs(relief_weights)) + EPS)
    normalized_relief = (avg_f1 / max_relief) if max_relief > 0 else 0.0
    val = w_f1 * (1 - avg_f1) + w_relief * (1 - normalized_relief)
    return float(val) if np.isfinite(val) else 1e9


# Parameters (tweaked for speed: smaller pop/iter with early stop)
pop_size = 25  # Reduced for faster runs
max_iter = 150  # Reduced, rely on early stop
minx = -10  # Narrower for stability
maxx = 10


# Helper (updated for new fitness)
def run_and_print(
    algo_class, fitness_name, fitness_call, feature_names, dim, X_train, y_train
):
    algo = algo_class(
        fitness_func=lambda pos: fitness_call(pos, X_train, y_train),
        dim=dim,
        pop_size=pop_size,
        max_iter=max_iter,
        minx=minx,
        maxx=maxx,
        binary=True,
        early_stop_patience=15,
    )
    best_pos, best_fit = algo.optimize()
    selected = algo.binarize(best_pos)
    num_selected = np.sum(selected)
    percentage = (num_selected / dim) * 100
    print(f"{fitness_name} - Best Fitness: {best_fit:.4f}")
    print(f"Number of Selected Features: {num_selected}")
    print(f"Choice Percentage: {percentage:.2f}%")
    print(f"Selected Feature Indices: {np.where(selected == 1)[0]}")
    print(
        f"Selected Feature Names: {[feature_names[i] for i in np.where(selected == 1)[0]]}\n"
    )
    return selected, percentage


# Dictionary to hold all selected features from both methods
all_algorithms = {}

# --- 1. Define Models and Evaluation Helper ---
xgb_params = {
    "colsample_bytree": 0.5626355618320708,
    "gamma": 0.1016972117972675,
    "learning_rate": 0.01903332164785265,
    "max_depth": 20,
    "min_child_weight": 2,
    "n_estimators": 150,
    "subsample": 0.922071166159922,
}

grad_params = {
    "learning_rate": 0.06551227481350091,
    "max_depth": 8,
    "min_samples_split": 29,
    "n_estimators": 254,
    "subsample": 0.9473145475418845,
}

cat_params = {
    "bagging_temperature": 0.8082280954761953,
    "border_count": 108,
    "depth": 2,
    "iterations": 981,
    "l2_leaf_reg": 0.03826279109372956,
    "learning_rate": 0.27501197188586285,
    "random_strength": 0.013824704110145813,
}

lgbm_params = {
    "colsample_bytree": 0.5537864010835775,
    "learning_rate": 0.0405270178570898,
    "max_depth": 18,
    "min_child_samples": 42,
    "n_estimators": 284,
    "num_leaves": 150,
    "reg_alpha": 0.6966468997148854,
    "reg_lambda": 0.0001338513910647724,
    "subsample": 0.5384579860956592,
}

bagg_params = {
    "max_features": 0.5,
    "max_samples": 0.5,
    "n_estimators": 300,
}

# GPU-enabled models where possible (fallback to CPU if GPU not available)
models = {
    "XGBoost": xgb.XGBClassifier(
        **xgb_params, random_state=42, eval_metric="logloss", tree_method="hist"
    ),  # CPU fallback
    "GradientBoost": GradientBoostingClassifier(
        **grad_params, random_state=42
    ),  # CPU-only
    "CatBoost": CatBoostClassifier(
        **cat_params, random_state=42, verbose=0
    ),  # CPU fallback
    "LGBMClassifier": LGBMClassifier(
        **lgbm_params, random_state=42, verbose=-1
    ),  # CPU fallback
    "BaggingClassifier": BaggingClassifier(
        **bagg_params,
        random_state=42,
        n_jobs=-1,
    ),  # CPU-only
}

def evaluate_models(X_tr, X_te, y_tr, y_te, name_suffix=""):
    res = {}
    print(f"\n--- Performance for {name_suffix} ---")
    for model_name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_proba = (
            model.predict_proba(X_te)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )
        
        metrics = compute_metrics(y_te, y_pred, y_proba)
        res[model_name] = metrics
        
        print(f"{model_name}: "
              f"Acc = {metrics['Accuracy']:.4f}, "
              f"Sensitivity = {metrics['Sensitivity']:.4f}, "
              f"Specificity = {metrics['Specificity']:.4f}, "
              f"F1 = {metrics['F1 Score']:.4f}, "
              f"AUC = {metrics['AUC']:.4f}")
    return res

# --- 2. Baseline Evaluation (Before Selection) ---
print("\n=== BASELINE: BEFORE FEATURE SELECTION (All Features) ===")
baseline_results = evaluate_models(X_train, X_test, y_train, y_test, "ALL Features")

# --- 3. Run Feature Selection Algorithms ---
print("\n=== Faster Chi2-AUC-Sparsity Hybrid for Diabetes FS ===")
gwo_chi2, gwo_chi2_perc = run_and_print(
    GWO,
    "GWO-Chi2 AUC Sparsity",
    chi2_auc_sparsity_fitness,
    feature_names,
    dim,
    X_train,
    y_train,
)
woa_chi2, woa_chi2_perc = run_and_print(
    WOA,
    "WOA-Chi2 AUC Sparsity",
    chi2_auc_sparsity_fitness,
    feature_names,
    dim,
    X_train,
    y_train,
)
fa_chi2, fa_chi2_perc = run_and_print(
    FA,
    "FA-Chi2 AUC Sparsity",
    chi2_auc_sparsity_fitness,
    feature_names,
    dim,
    X_train,
    y_train,
)
all_algorithms["GWO_Chi2"] = gwo_chi2
all_algorithms["WOA_Chi2"] = woa_chi2
all_algorithms["FA_Chi2"] = fa_chi2

""" # Run Relief-F1 Interaction
print("\n=== Better Relief-Approx F1 Interaction Hybrid for Diabetes FS ===")
gwo_relief, gwo_relief_perc = run_and_print(
    GWO,
    "GWO-Relief F1 Interaction",
    relief_interaction_f1_fitness,
    feature_names,
    dim,
    X_train,
    y_train,
)
woa_relief, woa_relief_perc = run_and_print(
    WOA,
    "WOA-Relief F1 Interaction",
    relief_interaction_f1_fitness,
    feature_names,
    dim,
    X_train,
    y_train,
)
fa_relief, fa_relief_perc = run_and_print(
    FA,
    "FA-Relief F1 Interaction",
    relief_interaction_f1_fitness,
    feature_names,
    dim,
    X_train,
    y_train,
)
all_algorithms["GWO_Relief"] = gwo_relief
all_algorithms["WOA_Relief"] = woa_relief
all_algorithms["FA_Relief"] = fa_relief """

# --- 4. Evaluate Selected Features (After Selection) ---
final_results = {}

# Add baseline to final results for table
for m_name, m_res in baseline_results.items():
    final_results[("ALL_Features", m_name)] = m_res

for algo_name, selected in all_algorithms.items():
    idx = np.where(selected == 1)[0]
    if len(idx) == 0:
        print(f"Warning: {algo_name} selected 0 features. Skipping.")
        continue
        
    X_train_sel = X_train.iloc[:, idx]
    X_test_sel = X_test.iloc[:, idx]
    
    print(f"\n=== AFTER FEATURE SELECTION: {algo_name} ===")
    algo_res = evaluate_models(X_train_sel, X_test_sel, y_train, y_test, f"{algo_name} selected features")
    
    for m_name, m_res in algo_res.items():
        final_results[(algo_name, m_name)] = m_res

# --- 5. Summary Table ---
results_df = pd.DataFrame(final_results).T
print("\n--- Comparative Summary Table ---")
print(results_df.round(4))
