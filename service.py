from __future__ import annotations
import bentoml
import asyncio

with bentoml.importing():
    import joblib
    import pandas as pd
    import numpy as np
    from typing import List
    import xgboost as xgb
    import lightgbm as lgbm
    from catboost import CatBoostClassifier

# --- Model Services (New Style: No Runners) ---
@bentoml.service(resources={"cpu": "1"})  # Minimal resources per model
class LgbmService:
    def __init__(self):
        self.model = bentoml.lightgbm.load_model(bentoml.models.get("lgbm_model"))

    @bentoml.api  # Internal API for depends() calls
    async def predict(self, input_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(input_df, num_iteration=self.model.best_iteration)


@bentoml.service(resources={"cpu": "1"})
class XgbService:
    def __init__(self):
        self.model = bentoml.xgboost.load_model(bentoml.models.get("xgb_model"))

    @bentoml.api
    async def predict(self, input_df: pd.DataFrame) -> np.ndarray:
        dmatrix = xgb.DMatrix(input_df)
        return self.model.predict(dmatrix)



@bentoml.service(resources={"cpu": "1"})
class CbService:
    def __init__(self):
        self.model = bentoml.catboost.load_model(bentoml.models.get("cb_model"))

    @bentoml.api
    async def predict(self, input_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(input_df)




# --- Ensemble Service ---
@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class DiabetesEnsembleService:
    # Inject model services as dependencies (local classes work fine)
    lgbm = bentoml.depends(LgbmService)
    xgb = bentoml.depends(XgbService)
    cb = bentoml.depends(CbService)

    def process_input(self, input_data: dict) -> pd.DataFrame:
        return pd.DataFrame([input_data])

    @bentoml.api(route="/predict/ensemble")
    async def predict_ensemble(self, input_data: dict) -> dict:
        input_df = self.process_input(input_data)
        # Concurrent predictions via gather
        lgbm_pred, xgb_pred, grad_pred, cb_pred, bag_pred = await asyncio.gather(
            self.lgbm.predict(input_df),
            self.xgb.predict(input_df),
            self.cb.predict(input_df),
        )
        predictions = [
            int(lgbm_pred[0]),
            int(xgb_pred[0]),
            int(cb_pred[0]),
        ]
        final_prediction = (
            max(set(predictions), key=predictions.count) if predictions else 0
        )
        return {
            "final_prediction": int(final_prediction),
            "individual_predictions": dict(
                zip(["lgbm", "xgb", "cb"], predictions)
            ),
        }

    @bentoml.api(route="/predict/lgbm")
    async def predict_lgbm(self, input_data: dict) -> dict:
        input_df = self.process_input(input_data)
        prediction = await self.lgbm.predict(input_df)
        return {"LGBM_Prediction": int(prediction[0])}

    @bentoml.api(route="/predict/xgb")
    async def predict_xgb(self, input_data: dict) -> dict:
        input_df = self.process_input(input_data)
        prediction = await self.xgb.predict(input_df)
        return {"XGB_Prediction": int(prediction[0])}

    @bentoml.api(route="/predict/cb")
    async def predict_cb(self, input_data: dict) -> dict:
        input_df = self.process_input(input_data)
        prediction = await self.cb.predict(input_df)
        return {"CatBoost_Prediction": int(prediction[0])}

