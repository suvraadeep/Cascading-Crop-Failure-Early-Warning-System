import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Optional
import sys
sys.path.append("..")
from config import cfg


def fetch_nasa_power(lat: float, lon: float,
                     days_back: int = 120) -> Optional[pd.DataFrame]:
    """
    Fetch recent weather from NASA POWER API.
    Completely free, no API key needed.
    """
    end   = datetime.today()
    start = end - timedelta(days=days_back)

    url = (
        f"{cfg.NASA_POWER_BASE}?"
        f"parameters={cfg.NASA_PARAMS}&community=AG"
        f"&longitude={lon:.4f}&latitude={lat:.4f}"
        f"&start={start.strftime('%Y%m%d')}&end={end.strftime('%Y%m%d')}"
        f"&format=JSON"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()["properties"]["parameter"]
        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df.columns = ["temp_c", "precip_mm", "humidity_pct",
                      "wind_ms", "solar_kwh", "dewpoint_c"]
        df = df.replace(-999.0, np.nan).interpolate()
        df = df.reset_index().rename(columns={"index": "date"})
        return df
    except Exception as e:
        print(f"NASA POWER error: {e}")
        return None


def engineer_inference_features(df: pd.DataFrame,
                                 soil_whc: float = 0.28,
                                 stress_sens: float = 0.9) -> pd.DataFrame:
    """Apply the same feature engineering as training."""
    df = df.copy().sort_values("date").reset_index(drop=True)
    t = np.arange(len(df))

    # Cyclical
    df["doy"]       = df["date"].dt.dayofyear
    df["doy_sin"]   = np.sin(2 * np.pi * df["doy"] / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * df["doy"] / 365)
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)

    # Rolling
    for win in [7, 14, 30]:
        df[f"precip_mm_roll{win}_sum"]  = df["precip_mm"].rolling(win, 1).sum()
        df[f"precip_mm_roll{win}_mean"] = df["precip_mm"].rolling(win, 1).mean()
        df[f"temp_c_roll{win}_mean"]    = df["temp_c"].rolling(win, 1).mean()
        df[f"temp_c_roll{win}_max"]     = df["temp_c"].rolling(win, 1).max()
        df[f"humidity_pct_roll{win}_mean"] = df["humidity_pct"].rolling(win, 1).mean()
        df[f"solar_kwh_roll{win}_mean"] = df["solar_kwh"].rolling(win, 1).mean()

    # VPD
    es = 0.6108 * np.exp(17.27 * df["temp_c"] / (df["temp_c"] + 237.3))
    ea = es * df["humidity_pct"] / 100
    df["vpd"] = np.clip(es - ea, 0, 5)

    # SPI30
    mu  = df["precip_mm"].mean()
    std = df["precip_mm"].std() + 1e-6
    df["spi30"] = ((df["precip_mm_roll30_mean"] - mu) / std).clip(-3, 3)

    # GDD
    df["gdd"]     = np.maximum(df["temp_c"] - 10, 0)
    df["gdd_cum"] = df["gdd"].cumsum() % 2000

    # Heat stress
    df["heat_stress"]    = (df["temp_c"] > 38).astype(float)
    df["heat_stress_7d"] = df["heat_stress"].rolling(7, 1).sum()

    # Water stress
    df["whc"]          = soil_whc
    df["water_stress"] = (1 - np.clip(df["precip_mm_roll14_sum"] / (soil_whc * 100 + 1e-6), 0, 1))

    # Days in season
    df["days_in_season"] = (df["doy"] % 180).astype(float)

    # Lags (using backfill for first rows)
    for lag in [1, 3, 7, 14]:
        df[f"temp_lag{lag}"]   = df["temp_c"].shift(lag).bfill()
        df[f"precip_lag{lag}"] = df["precip_mm"].shift(lag).fillna(0)

    df["stress_sens"] = stress_sens
    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    return df