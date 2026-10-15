import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class SeasonalNaive(RegressorMixin, BaseEstimator):
    def __init__(self, lag=24):
        self.lag = lag

    def fit(self, x, y):
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, x):
        return np.asarray(x[f"lag_{self.lag}"], dtype=float)

def candidates(seed=42, mode="full"):
    choices = {"seasonal_naive_24": SeasonalNaive(24), "seasonal_naive_168": SeasonalNaive(168)}
    for alpha in ([10.] if mode == "quick" else [1., 10., 100.]):
        choices[f"ridge_alpha{alpha:g}"] = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    choices["random_forest"] = RandomForestRegressor(
        n_estimators=50 if mode == "quick" else 200, min_samples_leaf=4,
        max_depth=18, random_state=seed, n_jobs=1)
    choices["hist_gradient_boosting"] = HistGradientBoostingRegressor(
        max_iter=70 if mode == "quick" else 220, max_leaf_nodes=20,
        learning_rate=.06, l2_regularization=1, early_stopping=False, random_state=seed)
    return choices

def metrics(y, prediction):
    return {"mae": mean_absolute_error(y, prediction),
            "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "r_squared": r2_score(y, prediction)}

def conformal_radius(residuals, alpha=.10):
    if not 0 < alpha < 1 or len(residuals) == 0:
        raise ValueError("Supply nonempty residuals and alpha strictly between 0 and 1.")
    residuals = np.sort(np.asarray(residuals, dtype=float))
    if not np.isfinite(residuals).all() or (residuals < 0).any():
        raise ValueError("Residuals must be finite and nonnegative.")
    rank = int(np.ceil((len(residuals) + 1) * (1 - alpha)))
    if rank > len(residuals):
        raise ValueError("Calibration sample is too small for the requested coverage.")
    return float(residuals[rank - 1])
