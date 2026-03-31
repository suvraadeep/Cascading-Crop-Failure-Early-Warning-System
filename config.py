import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

class Config:
    MODEL_PATH        = BASE_DIR / "models" / "tft_best.ckpt"
    CONFORMAL_PATH    = BASE_DIR / "models" / "conformal.pkl"
    FARM_METADATA     = BASE_DIR / "models" / "farm_metadata.csv"
    ENCODER_LENGTH    = 90
    PRED_LENGTH       = 21
    QUANTILES         = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    STRESS_THRESHOLD  = 0.35
    CONFORMAL_ALPHA   = 0.1

    NASA_POWER_BASE   = "https://power.larc.nasa.gov/api/temporal/daily/point"
    NASA_PARAMS       = "T2M,PRECTOTCORR,RH2M,WS2M,ALLSKY_SFC_SW_DWN,T2MDEW"

    TWILIO_SID        = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH       = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_FROM       = os.getenv("TWILIO_FROM_NUMBER", "")
    TWILIO_TO         = os.getenv("TWILIO_TO_NUMBER", "")

    GEE_PROJECT       = os.getenv("GEE_PROJECT_ID", "")
    USE_GEE           = os.getenv("USE_GEE", "false").lower() == "true"

    RISK_COLORS = {
        "LOW":      "#2ECC71",
        "MODERATE": "#F39C12",
        "HIGH":     "#E74C3C",
        "CRITICAL": "#8E44AD",
    }

    @staticmethod
    def get_risk_level(min_ndvi: float) -> str:
        if min_ndvi > 0.50:  return "LOW"
        if min_ndvi > 0.40:  return "MODERATE"
        if min_ndvi > 0.30:  return "HIGH"
        return "CRITICAL"

cfg = Config()