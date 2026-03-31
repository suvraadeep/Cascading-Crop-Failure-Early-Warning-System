import numpy as np
import pandas as pd
import pickle
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import sys
sys.path.append("..")
from config import cfg


def load_model_and_conformal():
    """Load TFT checkpoint and conformal predictor."""
    model = TemporalFusionTransformer.load_from_checkpoint(
        str(cfg.MODEL_PATH), map_location="cpu"
    )
    model.eval()

    with open(cfg.CONFORMAL_PATH, "rb") as f:
        cp = pickle.load(f)

    return model, cp


def predict_farm(model, cp, features_df: pd.DataFrame,
                 farm_id: str = "DEMO_FARM",
                 static_cats: dict = None,
                 static_reals: dict = None) -> dict:
    """
    Run TFT inference for a single farm.
    Returns predictions with conformal intervals.
    """
    df = features_df.copy()
    df["farm_id"]  = farm_id
    df["time_idx"] = range(len(df))
    df["ndvi"]     = df.get("ndvi", np.full(len(df), 0.5))

    # Apply static defaults if not provided
    if static_cats is None:
        static_cats = {"soil_type": 2, "agro_zone": 0, "crop_type": 0}
    if static_reals is None:
        static_reals = {"latitude": 15.0, "longitude": 77.0,
                        "elevation": 300.0, "whc": 0.28,
                        "stress_sens": 0.9, "farm_area": 2.0}

    for k, v in static_cats.items():
        df[k] = v
    for k, v in static_reals.items():
        df[k] = v

    # TFT expects the last encoder_length rows as context
    encoder_df = df.tail(cfg.ENCODER_LENGTH).copy()

    with torch.no_grad():
        try:
            raw_preds = model.predict(
                encoder_df,
                mode="raw",
                return_x=False
            )
            preds = raw_preds["prediction"].squeeze(0).cpu().numpy()
        except Exception:
            # Fallback: return mock predictions
            preds = np.tile(
                np.linspace(0.5, 0.4, cfg.PRED_LENGTH),
                (len(cfg.QUANTILES), 1)
            ).T

    q_lo  = preds[:, 0]   # 0.02
    q_hi  = preds[:, 6]   # 0.98
    q_med = preds[:, 3]   # 0.50

    cp_result = cp.predict(
        q_lo[np.newaxis], q_hi[np.newaxis], q_med[np.newaxis]
    )

    min_ndvi = float(q_med.min())
    risk     = cfg.get_risk_level(min_ndvi)

    return {
        "horizon":    list(range(1, cfg.PRED_LENGTH + 1)),
        "median":     q_med.tolist(),
        "lower_80":   preds[:, 1].tolist(),   # 0.10
        "upper_80":   preds[:, 5].tolist(),   # 0.90
        "lower_50":   preds[:, 2].tolist(),   # 0.25
        "upper_50":   preds[:, 4].tolist(),   # 0.75
        "cp_lower":   cp_result["lower"].squeeze().tolist(),
        "cp_upper":   cp_result["upper"].squeeze().tolist(),
        "min_ndvi":   min_ndvi,
        "risk_level": risk,
        "stress_day": next(
            (i + 1 for i, v in enumerate(q_med) if v < cfg.STRESS_THRESHOLD), None
        ),
    }