import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from threadpoolctl import threadpool_limits
from .data import FEATURES, load_data, obtain_data, partition, next_features
from .models import candidates, metrics, conformal_radius
from .common import output_dirs, provenance, save_json, html_report, dataset_label

@threadpool_limits.wrap(limits=1)
def train(mode="full", seed=42, data=None, output="outputs/full"):
    source = Path(data) if data else obtain_data()
    dataset = dataset_label(source, "UCI Bike Sharing hour.csv", supplied=data is not None)
    frame, data_audit = load_data(source, mode)
    splits = partition(frame)
    out = output_dirs(output)
    pd.concat([pd.DataFrame({"timestamp": part.index, "split": name})
               for name, part in splits.items()]).to_csv(out / "audit/split_registry.csv", index=False)
    fitted, validation = {}, []
    tr, va, ca, te = (splits[name] for name in ("train", "validation", "calibration", "test"))
    for name, model in candidates(seed, mode).items():
        model.fit(tr[FEATURES], tr.target)
        prediction = np.maximum(0, model.predict(va[FEATURES]))
        validation.append({"model": name, "validation_mae": metrics(va.target, prediction)["mae"]})
        fitted[name] = model
    leaderboard = pd.DataFrame(validation).sort_values("validation_mae", kind="stable")
    selected = str(leaderboard.iloc[0].model)
    model = fitted[selected]
    cv = []
    for fold, (fit_ids, holdout_ids) in enumerate(TimeSeriesSplit(n_splits=3).split(tr), start=1):
        fold_model = clone(model).fit(tr[FEATURES].iloc[fit_ids], tr.target.iloc[fit_ids])
        p = np.maximum(0, fold_model.predict(tr[FEATURES].iloc[holdout_ids]))
        cv.append({"fold": fold, "training_rows": len(fit_ids), "validation_rows": len(holdout_ids),
                   "last_training_time": str(tr.index[fit_ids[-1]]),
                   "first_validation_time": str(tr.index[holdout_ids[0]]),
                   **metrics(tr.target.iloc[holdout_ids], p)})
    radius = conformal_radius(np.abs(ca.target - np.maximum(0, model.predict(ca[FEATURES]))))
    artifact = {"model": model, "features": FEATURES, "selected": selected,
                "radius": radius, "nominal_coverage": .90, "horizon_hours": 1,
                "training_data_source": dataset}
    joblib.dump(artifact, out / "models/model.joblib")
    rows = [{"model": name, "selected_from_validation": name == selected,
             **metrics(te.target, np.maximum(0, candidate.predict(te[FEATURES])))}
            for name, candidate in fitted.items()]
    measured = pd.DataFrame(rows)
    measured["data_source"] = dataset
    measured.to_csv(out / "reports/metrics.csv", index=False)
    leaderboard.to_csv(out / "reports/validation.csv", index=False)
    cv_frame = pd.DataFrame(cv)
    cv_frame.to_csv(out / "reports/expanding_window_diagnostics.csv", index=False)
    prediction = np.maximum(0, model.predict(te[FEATURES]))
    lower, upper = np.maximum(0, prediction - radius), prediction + radius
    interval = {"nominal_coverage": .90, "test_empirical_coverage": float(
                    np.mean((te.target >= lower) & (te.target <= upper))),
                "test_mean_width": float(np.mean(upper - lower)), "calibration_radius": radius,
                "interpretation": "Empirical interval diagnostic; temporal exchangeability is not assumed."}
    save_json(out / "reports/interval_metrics.json", interval)
    pd.DataFrame({"timestamp": te.index, "target": te.target, "prediction": prediction,
                  "lower_90": lower, "upper_90": upper}).to_csv(
                      out / "reports/test_predictions.csv", index=False)
    count = min(len(te), 168)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(te.index[-count:], te.target.iloc[-count:], label="observed", lw=1.5)
    ax.plot(te.index[-count:], prediction[-count:], label="one-hour forecast", lw=1.2)
    ax.fill_between(te.index[-count:], lower[-count:], upper[-count:], alpha=.2, label="interval")
    ax.set(ylabel="Bike rentals", xlabel="Target hour", title=f"Final {count} test forecasts")
    if dataset.startswith("SIMULATED"):
        ax.set_title(f"Final {count} test forecasts — simulated software demonstration")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "reports/forecast.png", dpi=150)
    plt.close(fig)
    record = provenance(mode, seed, source)
    record.update({"dataset": dataset,
                   "data_audit": data_audit, "selected": selected,
                   "selection": "minimum validation MAE", "horizon": "rolling one-hour ahead",
                   "splits": {name: {"rows": len(part), "start": str(part.index.min()),
                                    "end": str(part.index.max())} for name, part in splits.items()},
                   "excluded": ["casual", "registered", "current-hour cnt", "current-hour weather"]})
    save_json(out / "audit/run.json", record)
    html_report(out / "reports/index.html", "One-hour-ahead bike demand",
                f"Data: {dataset}. Selected model: {selected}. Calendar is known at the target hour; "
                "counts and weather come from earlier hours. Evaluation rolls forward "
                "one hour at a time as actual observations become available.",
                [("Held-out test metrics", measured), ("Validation selection", leaderboard),
                 ("Expanding-window training diagnostics", cv_frame),
                 ("Interval diagnostic", pd.DataFrame([interval]))],
                [("forecast.png", "Observed counts, forecasts, and calibrated residual intervals.")])
    print(measured.to_string(index=False))
    print(f"Report: {out / 'reports/index.html'}")
    return out

def predict(model_path, input_path=None, history_path=None):
    artifact = joblib.load(model_path)
    if history_path:
        timestamp, x = next_features(history_path)
    else:
        values = json.loads(Path(input_path).read_text())
        x = pd.DataFrame([values] if isinstance(values, dict) else values)
        timestamp = None
        if set(FEATURES) - set(x.columns):
            raise ValueError("The prepared input is missing required features.")
        x = x[FEATURES].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(x.to_numpy()).all():
            raise ValueError("Features must be finite.")
    p = np.maximum(0, artifact["model"].predict(x))
    return [{"timestamp": str(timestamp) if timestamp is not None else None,
             "forecast_rentals": float(value), "lower_90": float(max(0, value - artifact["radius"])),
             "upper_90": float(value + artifact["radius"]), "horizon_hours": 1,
             "training_data_source": artifact["training_data_source"]}
            for value in p]
