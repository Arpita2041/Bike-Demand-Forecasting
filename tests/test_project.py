import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from project.data import hourly_frame, feature_frame, FEATURES, partition, next_features
from project.models import conformal_radius
from project.workflow import train, predict

def fixture(n=1400):
    """Simulated hourly observations; not UCI bike demand measurements."""
    index = pd.date_range("2011-01-01", periods=n, freq="h")
    rng = np.random.default_rng(7)
    count = 120 + 80 * np.sin(2 * np.pi * index.hour / 24) + rng.normal(0, 8, n)
    return pd.DataFrame({"dteday": index.strftime("%Y-%m-%d"), "hr": index.hour,
                         "cnt": np.maximum(0, count).astype(int),
                         "temp": .5 + .2 * np.sin(np.arange(n) / 120),
                         "hum": .6, "windspeed": .15, "weathersit": 1,
                         "casual": 999, "registered": 999})

class ForecastTests(unittest.TestCase):
    def test_features_do_not_use_target_or_future_observations(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hours.csv"
            fixture().to_csv(path, index=False)
            raw = hourly_frame(path)
            timestamp = raw.index[400]
            expected = feature_frame(raw).loc[timestamp]
            changed = raw.copy()
            changed.loc[timestamp:, ["cnt", "temp", "hum", "weathersit", "windspeed"]] += 9999
            pd.testing.assert_series_equal(feature_frame(changed).loc[timestamp], expected)
            self.assertNotIn("casual", FEATURES)
            self.assertNotIn("registered", FEATURES)

    def test_lags_are_calendar_hours_and_gaps_are_not_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hours.csv"
            fixture().drop(index=300).to_csv(path, index=False)
            raw = hourly_frame(path)
            self.assertTrue(pd.isna(raw.cnt.iloc[300]))
            self.assertTrue(pd.isna(feature_frame(raw).lag_24.iloc[324]))
            self.assertEqual(feature_frame(raw).lag_168.iloc[400], raw.cnt.iloc[232])

    def test_quantile_rank_and_chronology(self):
        self.assertEqual(conformal_radius(np.arange(100)), 90.)
        with self.assertRaises(ValueError):
            conformal_radius([1., 2.])
        frame = pd.DataFrame(index=pd.date_range("2011-01-01", periods=1000, freq="h"))
        parts = partition(frame)
        last = pd.Timestamp("2010-01-01")
        for part in parts.values():
            self.assertGreater(part.index.min(), last)
            last = part.index.max()

    def test_complete_fixture_run_and_next_hour(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, output = Path(temporary) / "hours.csv", Path(temporary) / "run"
            fixture().to_csv(path, index=False)
            train("quick", 42, path, output)
            timestamp, x = next_features(path)
            self.assertEqual(timestamp, pd.Timestamp("2011-01-01") + pd.Timedelta(hours=1400))
            self.assertEqual(x.lag_1.iloc[0], fixture().cnt.iloc[-1])
            result = predict(output / "models/model.joblib", history_path=path)
            self.assertEqual(result[0]["horizon_hours"], 1)
            self.assertLessEqual(result[0]["lower_90"], result[0]["upper_90"])
            self.assertTrue((output / "reports/interval_metrics.json").exists())

if __name__ == "__main__":
    unittest.main()
