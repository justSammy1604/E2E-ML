import mlflow
import os

# --- DagsHub credentials ---
username = "justSammy1604"
repo_name = "Diabetes-Models"
token = '390424ddda01a86ea614e46214ab76ac27716817'

# --- Map your run IDs to model names ---
models = {
    # "78957450d3c74f5d83ef09e5b134f001": "LGBM",
    # "b0b0c8436b9f46658979827f108d62be": "XGBoost",
    # "818e3a3768744ed7acb232ffa729b8e3": "GradBoost",
    "980a1684be5a437b9d25c199eb4f4aa9": "CatBoost",
    # "c53657c6a89c44609fae8f76c4795929": "Bagging"
}

# --- Authenticate MLflow with DagsHub ---
mlflow.set_tracking_uri(
    f"https://{username}:{token}@dagshub.com/{username}/{repo_name}.mlflow"
)

# --- Local destination directory ---
base_dir = os.path.join("Diabetes-Models", "models")
os.makedirs(base_dir, exist_ok=True)

# --- Download all models ---
for run_id, model_name in models.items():
    dst_path = os.path.join(base_dir, model_name)
    os.makedirs(dst_path, exist_ok=True)

    print(f"⬇️  Downloading {model_name} model from run {run_id} ...")
    try:
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model/model.cb",  # path in DagsHub
            dst_path=dst_path,  # local save path
        )
        print(f"✅ {model_name} model downloaded successfully to {dst_path}\n")
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}\n")
