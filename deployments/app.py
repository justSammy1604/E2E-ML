import os
import mlflow
import dagshub
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
import mlflow.sklearn
import mlflow.xgboost
import mlflow.catboost 
import xgboost as xgb
import catboost as cb
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
import numpy as np
from src.feat_scale import X_train_scaled, X_test_scaled, y_train, y_test

# -----------------------------
# 1️⃣ Load environment variables
# -----------------------------
load_dotenv()  # loads .env from project root

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

if not all([MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD]):
    raise EnvironmentError(
        "Missing one or more MLflow environment variables. "
        "Please check your .env file."
    )

# Apply credentials for MLflow
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

# -----------------------------
# 2️⃣ Initialize DagsHub + MLflow
# -----------------------------
dagshub.init(repo_owner="justSammy1604", repo_name="Diabetes-Models", mlflow=True)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Diabetes Prediction Models")

print(f"Connected to DagsHub MLflow at {MLFLOW_TRACKING_URI}")

# -----------------------------
# 3️⃣ Model parameters
# -----------------------------
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

models = [
    ("XGBoost", xgb_params, xgb.XGBClassifier()),
    ("GradientBoosting", grad_params, GradientBoostingClassifier()),
    ("CatBoost", cat_params, cb.CatBoostClassifier(verbose=0)),
    ("LightGBM", lgbm_params, LGBMClassifier()),
    ("BaggingClassifier", bagg_params, BaggingClassifier()),
]

# -----------------------------
# 4️⃣ Training + Logging loop
# -----------------------------
for model_name, params, model in models:
    print(f"\n🔹 Training {model_name}...")

    # Ensure labels are numpy 1D arrays
    y_train_np = np.asarray(y_train).ravel()
    y_test_np = np.asarray(y_test).ravel()

    model.set_params(**params)
    model.fit(X_train_scaled, y_train_np)
    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test_np, y_pred, output_dict=True)

    # Safely close any active run before starting new one
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run()
    except Exception:
        pass

    with mlflow.start_run(run_name=model_name):
        # Log model
        if model_name == "XGBoost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        elif model_name == "CatBoost":
            mlflow.catboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        # Log params and metrics
        mlflow.log_params(params)
        acc = float(report.get("accuracy", 0.0))
        weighted = report.get("weighted avg", {})

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_weighted", float(weighted.get("precision", 0.0)))
        mlflow.log_metric("recall_weighted", float(weighted.get("recall", 0.0)))
        mlflow.log_metric("f1_weighted", float(weighted.get("f1-score", 0.0)))

    print(f"✅ {model_name} logged to DagsHub successfully.")

print("\n🎉 All models trained and logged to DagsHub MLflow!")
