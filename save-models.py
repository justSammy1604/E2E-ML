import bentoml
import joblib
from bentoml.io import JSON
import pandas as pd
import numpy as np
from typing import List
import xgboost as xgb
import lightgbm as lgbm
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier

# Import the specific class for clarity

# --- 1. Define File Paths ---
MODEL_PATHS = {
    "lgbm": "Diabetes-Models/models/LGBM/model/model.pkl",
    "xgb": "Diabetes-Models/models/XGBoost/model/model.xgb",
    "grad": "Diabetes-Models/models/GradBoost/model/model.pkl",
    "cb": "Diabetes-Models/models/CatBoost/model/model.cb",
    "bag": "Diabetes-Models/models/Bagging/model/model.pkl",
}

# --- 2. Load Models from Files ---
print("Loading models from file system...")

# Load LGBM (joblib/pickle)
lgbm_model = joblib.load(MODEL_PATHS["lgbm"])

# Load XGBoost (native Booster load, fixed from previous step)
xgb_model_booster = xgb.Booster()
xgb_model_booster.load_model(MODEL_PATHS["xgb"])

# Load Scikit-learn models (joblib/pickle)
grad_model = joblib.load(MODEL_PATHS["grad"])
bag_model = joblib.load(MODEL_PATHS["bag"])

# Load CatBoost (CatBoost models are often saved natively or with CatBoost's specific methods)
# Assuming joblib was used, but CatBoostClassifier().load_model() might also be necessary.
try:
    cb_model = joblib.load(MODEL_PATHS["cb"])
except:
    # Fallback for native CatBoost loading if joblib fails
    cb_model = CatBoostClassifier()
    cb_model.load_model(MODEL_PATHS["cb"])


# --- 3. Save Models to BentoML Model Store ---
print("Saving models to BentoML Model Store...")

bentoml.lightgbm.save_model("lgbm_model", lgbm_model)
bentoml.xgboost.save_model("xgb_model", xgb_model_booster)
bentoml.sklearn.save_model("grad_model", grad_model)
bentoml.catboost.save_model("cb_model", cb_model)
bentoml.sklearn.save_model("bag_model", bag_model)

print("All models successfully saved to the BentoML Model Store.")