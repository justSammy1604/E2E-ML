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

# Diabetes Risk Prediction: End-to-End ML Pipeline

This repository contains a comprehensive, end-to-end machine learning pipeline for predicting diabetes risk. It spans from algorithmic research in feature selection to production-grade model deployment using modern MLOps practices.

## 🚀 Overview

The project aims to provide a robust and scalable solution for diabetes risk assessment. It features:
- **Custom Metaheuristic Optimization:** Implementation of nature-inspired algorithms for intelligent feature selection.
- **Advanced Ensemble Modeling:** High-performance classification using state-of-the-art gradient boosting frameworks.
- **Robust MLOps Pipeline:** Seamless experiment tracking, data versioning, and model serving.
- **Interactive Web Interface:** A user-friendly frontend for real-time risk prediction.

## ✨ Key Features

### 1. Nature-Inspired Feature Selection
Implementation of three powerful metaheuristic algorithms to identify the most predictive health indicators:
- **Grey Wolf Optimizer (GWO)**
- **Whale Optimization Algorithm (WOA)**
- **Firefly Algorithm (FA)**
- *Hybrid Fitness Functions:* Utilizing Chi2-AUC-Sparsity metrics to optimize for both accuracy and model parsimony.

### 2. High-Performance Modeling
- **Ensemble Approach:** Combines **XGBoost**, **LightGBM**, and **CatBoost** via a voting ensemble to improve generalization and robustness.
- **Imbalance Handling:** Utilizes **SMOTE** and **ADASYN** to address class distribution skew in health data.
- **Hyperparameter Optimization:** Leverages **Optuna** for Bayesian optimization of model parameters.

### 3. MLOps & Engineering
- **Experiment Tracking:** Integrated with **MLFlow** and **DagsHub** for comprehensive logging of metrics, parameters, and model artifacts.
- **Data Versioning:** Uses **DVC** to ensure data reproducibility and lineage.
- **Model Serving:** Built with **BentoML** to provide a high-performance, asynchronous microservice architecture.
- **Dependency Management:** Managed via **uv** for fast and reliable environment setup.

## 🛠️ Tech Stack

- **Languages:** Python 3.12+, JavaScript/HTML/CSS
- **ML Frameworks:** Scikit-learn, XGBoost, LightGBM, CatBoost, Imbalanced-learn
- **Optimization:** Optuna, Custom Metaheuristic Implementations
- **MLOps Tools:** MLFlow, DagsHub, DVC, BentoML
- **Data Processing:** Pandas, Polars, NumPy
- **Frontend:** HTML5, CSS3, Vanilla JavaScript (Fetch API)

## 📁 Project Structure

```text
├── .dvc/               # DVC configuration and cache
├── data/                # Data directory (managed by DVC)
├── deployments/         # MLFlow and DagsHub integration scripts
├── frontend/            # Web interface for prediction
├── src/                 # Core source code
│   ├── fa.py            # Firefly Algorithm
│   ├── gwo.py           # Grey Wolf Optimizer
│   ├── woa.py           # Whale Optimization Algorithm
│   ├── feat_select.py   # Feature selection pipeline
│   ├── feat_scale.py    # Data scaling and preprocessing
│   └── metrics.py       # Custom evaluation metrics
├── service.py           # BentoML service definition
├── save-models.py       # Script to save trained models to BentoML store
├── bentofile.yaml       # BentoML build configuration
├── pyproject.toml       # Project dependencies and metadata
└── README.md            # Project documentation
```

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install dependencies using `uv`:**
   ```bash
   uv sync
   ```

3. **Initialize DVC:**
   ```bash
   dvc pull
   ```

## 🖥️ Usage

### Training and Feature Selection
To run the feature selection and baseline evaluation:
```bash
python src/feat_select.py
```

### Experiment Tracking
To log models and metrics to DagsHub/MLFlow:
```bash
python deployments/app.py
```

### Model Serving
1. **Save models to BentoML store:**
   ```bash
   python save-models.py
   ```

2. **Serve the model locally:**
   ```bash
   bentoml serve service.py:DiabetesEnsembleService --reload
   ```

3. **Access the Frontend:**
   Open `frontend/index.html` in your browser. Ensure the BentoML service is running on `http://localhost:3000`.

## 📊 Results

The ensemble model achieves a balance of performance metrics across various feature selection strategies:
- **Accuracy:** ~89%
- **AUC:** ~0.95
- **F1-Score:** ~0.86

*For detailed comparative analysis, refer to the logs in MLFlow or the summary tables generated by `feat_select.py`.*
