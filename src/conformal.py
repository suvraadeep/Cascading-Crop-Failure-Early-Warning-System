import numpy as np
import pickle
from typing import Dict


class ConformalPredictor:
    """Inductive Conformal Predictor for marginal coverage guarantees."""

    def __init__(self, alpha: float = 0.1):
        self.alpha   = alpha
        self.q_hat   = None

    def calibrate(self, y_true: np.ndarray,
                  q_lo: np.ndarray,
                  q_hi: np.ndarray) -> "ConformalPredictor":
        nonconformity = np.maximum(q_lo - y_true, y_true - q_hi)
        n     = len(nonconformity)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(nonconformity, min(level, 1.0), axis=0)
        return self

    def predict(self, q_lo: np.ndarray,
                q_hi: np.ndarray,
                q_median: np.ndarray) -> Dict:
        assert self.q_hat is not None, "Call calibrate() first"
        return {
            "lower":  np.clip(q_lo  - self.q_hat, 0, 1),
            "median": q_median,
            "upper":  np.clip(q_hi  + self.q_hat, 0, 1),
            "width":  (q_hi + self.q_hat) - (q_lo - self.q_hat),
        }

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "ConformalPredictor":
        with open(path, "rb") as f:
            return pickle.load(f)