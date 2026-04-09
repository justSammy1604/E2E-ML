# End-to-End Diabetes Prediction Pipeline (E2E-ML)

An industry-grade machine learning project focused on diabetes risk prediction. This project implements a full lifecycle—from data preprocessing and feature selection using meta-heuristic algorithms to high-performance model serving with an ensemble architecture.

## 🚀 Overview

This repository demonstrates a robust Data Science workflow for a binary classification task: predicting diabetes based on health indicators. The project goes beyond simple model training by incorporating advanced feature selection techniques and production-ready serving layers.

## 🛠️ Tech Stack

### Data Science & Modeling
*   **Core:** `Python`, `Numpy`, `Pandas`, `Polars`
*   **Machine Learning:** `Scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost`
*   **Deep Learning:** `TensorFlow`/`Keras` (CNN, RNN, DNN implementations in `baseline/`)
*   **Imbalanced Data:** `imbalanced-learn` (SMOTE, ADASYN)
*   **Optimization:** `Optuna` (Hyperparameter Tuning)

### Feature Selection (Meta-heuristics)
*   `Grey Wolf Optimizer (GWO)`
*   `Whale Optimization Algorithm (WOA)`
*   `Firefly Algorithm (FA)`

### MLOps & Deployment
*   **Experiment Tracking:** `MLflow`
*   **Data Versioning:** `DVC`
*   **Model Serving:** `BentoML`
*   **API Framework:** `FastAPI`
*   **Environment:** `UV`, `Pip`

---

## 🧬 Project Architecture

### 1. Data Engineering & Preprocessing
The `data/` directory contains various stages of the dataset:
*   `diabetes_cleaned.csv`: Initial cleaning and normalization.
*   `diabetes_smote.csv` / `diabetes_adasyn.csv`: Synthetic data sampling to handle class imbalance.
*   `diabetes_complete_clean.csv.dvc`: Managed via DVC for version control.

### 2. Advanced Feature Selection
Located in `src/`, this project implements a hybrid approach to find the most predictive features while maintaining model sparsity:
*   **Objective Function:** A custom hybrid score combining `Chi-squared` statistics, `ROC-AUC`, and a `Sparsity` penalty.
*   **Optimizers:** Implementation of GWO, WOA, and FA to navigate the high-dimensional feature search space.

### 3. Model Training & Baselines
*   `baseline/`: A comprehensive library of model implementations including tree ensembles, gradient boosting, and deep learning architectures.
*   `trials/`: Iterative experiments for model refinement.
*   `mlruns/`: Integrated MLflow tracking for hyperparameter optimization and performance metrics.

### 4. Production Serving (MLOps)
The project utilizes **BentoML** for a robust serving layer:
*   `service.py`: Defines a high-performance `DiabetesEnsembleService`.
*   **Ensemble Strategy:** Combines `LightGBM`, `XGBoost`, and `CatBoost` using asynchronous prediction calls via `asyncio.gather` for minimal latency.
*   **CORS Support:** Integrated for seamless frontend communication.


---

##  Getting Started

### Installation
Ensure you have `uv` or `pip` installed:
```bash
pip install .
```

### 1. Model Preparation
First, train and save your models. Then, move them into the BentoML model store:
```bash
python save-models.py
```

### 2. Serving the Models
Start the production-ready API server:
```bash
bentoml serve service:DiabetesEnsembleService --reload
```

### 3. Frontend Access
Open `frontend/index.html` in your browser to interact with the API.

---

## 📊 Key Results
The ensemble approach using GWO-selected features achieved:
*   **Accuracy:** ~89.1%
*   **ROC-AUC:** ~0.946
*   **Sensitivity:** ~76.4%
*   **Specificity:** ~97.9%

*Refer to `src/metrics.py` comments for detailed feature importance and selection results.*
