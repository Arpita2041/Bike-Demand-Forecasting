"""Exact hourly lags with past weather and known target-hour calendar."""
import numpy as np
import pandas as pd
from .common import ROOT, download_member

URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
FEATURES = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin",
            "month_cos", "trend_days", "lag_1", "lag_24", "lag_168",
            "mean_24", "std_24", "observations_24", "mean_168",
            "past_temp", "past_humidity", "past_windspeed", "past_weather"]

def obtain_data():
    return download_member(URL, ROOT / "data/raw/hour.csv", "hour.csv")

def hourly_frame(path):
    frame = pd.read_csv(path)
    required = ["dteday", "hr", "cnt", "temp", "hum", "windspeed", "weathersit"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing raw hourly columns: {missing}")
    for name in required[1:]:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if not frame.hr.between(0, 23).all() or not np.equal(frame.hr, np.floor(frame.hr)).all():
        raise ValueError("hr must be an integer between 0 and 23.")
    if not np.isfinite(frame[required[1:]].to_numpy()).all() or (frame.cnt < 0).any():
        raise ValueError("The observed input rows must be finite with nonnegative counts.")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.dteday) + pd.to_timedelta(frame.hr, unit="h"))
    if frame.index.has_duplicates:
        raise ValueError("Duplicate hourly timestamps are not allowed.")
    frame = frame.sort_index()
    complete = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
    return frame.reindex(complete)

def feature_frame(raw):
    index = raw.index
    f = pd.DataFrame(index=index)
    for name, values, period in (("hour", index.hour, 24), ("weekday", index.dayofweek, 7),
                                ("month", index.month - 1, 12)):
        f[name + "_sin"] = np.sin(2 * np.pi * values / period)
        f[name + "_cos"] = np.cos(2 * np.pi * values / period)
    f["trend_days"] = (index - pd.Timestamp("2011-01-01")).total_seconds() / 86400
    for lag in (1, 24, 168):
        f[f"lag_{lag}"] = raw.cnt.shift(lag)
    past = raw.cnt.shift(1)
    f["mean_24"] = past.rolling(24, min_periods=18).mean()
    f["std_24"] = past.rolling(24, min_periods=18).std()
    f["observations_24"] = past.rolling(24).count()
    f["mean_168"] = past.rolling(168, min_periods=144).mean()
    for feature, source in (("past_temp", "temp"), ("past_humidity", "hum"),
                            ("past_windspeed", "windspeed"), ("past_weather", "weathersit")):
        f[feature] = raw[source].shift(1)
    return f[FEATURES]

def load_data(path, mode="full"):
    raw = hourly_frame(path)
    original_hours, missing = len(raw), int(raw.cnt.isna().sum())
    if mode == "quick" and len(raw) > 9000:
        raw = raw.iloc[-9000:]
    f = feature_frame(raw)
    f["target"] = raw.cnt
    data = f.dropna().copy()
    if len(data) < 400:
        raise ValueError("At least 400 complete forecast examples are required after lag construction.")
    return data, {"calendar_hours": original_hours, "missing_observed_hours": missing,
                  "usable_forecast_rows": len(data),
                  "first_target_time": str(data.index.min()),
                  "last_target_time": str(data.index.max())}

def partition(data):
    cuts = [0, int(len(data) * .60), int(len(data) * .75), int(len(data) * .85), len(data)]
    names = ["train", "validation", "calibration", "test"]
    return {name: data.iloc[a:b].copy()
            for name, a, b in zip(names, cuts[:-1], cuts[1:])}

def next_features(history_path):
    raw = hourly_frame(history_path)
    timestamp = raw.index.max() + pd.Timedelta(hours=1)
    raw = raw.reindex(pd.date_range(raw.index.min(), timestamp, freq="h"))
    x = feature_frame(raw).loc[[timestamp]]
    if x.isna().any().any():
        raise ValueError("History does not contain the required lag hours and rolling-window observations.")
    return timestamp, x
