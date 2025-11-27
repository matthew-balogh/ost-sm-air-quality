import math
from collections import deque
from datetime import datetime
from threading import Lock

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from listeners.sliding_window_listener import SlidingWindowListener


class _QuantileSketch:
    """
    Streaming quantile sketch without external deps.
    Keeps a bounded buffer and uses numpy quantiles; good enough for drift checks.
    """

    def __init__(self, maxlen=5000):
        self._buf = deque(maxlen=maxlen)

    def update(self, value):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return
        self._buf.append(float(value))

    def quantile(self, q):
        if not self._buf:
            return None
        return float(np.quantile(self._buf, q))


class OnlineForecaster(SlidingWindowListener):
    """
    Online forecaster that incrementally adapts to new data instead of relying
    solely on once-trained artifacts. It keeps short sliding windows for
    local anomaly checks, maintains long-horizon quantile sketches (TDigest
    when installed) to detect drift, and trains lightweight SGD regressors
    on-the-fly using lag/time features. Buffers stay bounded to avoid storing
    the full history; forecasts and anomalies can be written to InfluxDB
    via the provided dbWriter.
    """

    TOPIC_TO_COLUMN = {
        "pt08_s1_co": "PT08.S1(CO)",
        "pt08_s2_nmhc": "PT08.S2(NMHC)",
        "pt08_s3_nox": "PT08.S3(NOx)",
        "pt08_s4_no2": "PT08.S4(NO2)",
        "pt08_s5_o3": "PT08.S5(O3)",
        "t": "T",
        "rh": "RH",
        "ah": "AH",
    }
    COLUMN_TO_TOPIC = {v: k for k, v in TOPIC_TO_COLUMN.items()}

    def __init__(
        self,
        horizons=(1,),
        lag_hours=(1, 2, 3, 6),
        roll_windows=(3, 6),
        max_buffer_size=200,
        short_window=6,
        quantile_q=0.995,
        z_threshold=3.5,
        min_train_samples=1,
        dbWriter=None,
        verb=False,
    ):
        super().__init__()
        self.horizons = tuple(sorted(set(horizons)))
        self.lag_hours = tuple(lag_hours)
        self.roll_windows = tuple(roll_windows)
        self.max_buffer_size = max_buffer_size
        self.short_window = short_window
        self.quantile_q = quantile_q
        self.z_threshold = z_threshold
        self.dbWriter = dbWriter
        self.verb = verb
        self.min_train_samples = min_train_samples

        self.data_buffer = {topic: deque(maxlen=max_buffer_size) for topic in self.TOPIC_TO_COLUMN}
        self.buffer_lock = Lock()

        self.quantiles = {topic: _QuantileSketch() for topic in self.TOPIC_TO_COLUMN}
        self.models = {}
        self.model_initialized = {}
        self.pending_feature = {}  # last feature row awaiting the next actual value
        self.last_predictions = {}
        self.scalers = {}  # target_name -> StandardScaler
        self.train_counts = {}

    # --- core helpers -----------------------------------------------------
    def _normalize_value(self, value):
        if value == -200 or value == "-200":
            return np.nan
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    def _update_buffer(self, topic, value):
        val = self._normalize_value(value)
        with self.buffer_lock:
            self.data_buffer.setdefault(topic, deque(maxlen=self.max_buffer_size)).append(val)
        self.quantiles[topic].update(val)
        return val

    def _extract_features_from_series(self, values, extra_series=None):
        required = max(self.lag_hours) + 1
        if len(values) < required:
            return None
        values = values[-required:]
        now = datetime.now()
        idx = pd.date_range(end=now, periods=required, freq="h")
        df = pd.DataFrame({"target": values}, index=idx)

        feature_dict = {}
        for L in self.lag_hours:
            feature_dict[f"lag{L}"] = df["target"].shift(L)
        for W in self.roll_windows:
            feature_dict[f"rollmean{W}"] = df["target"].shift(1).rolling(W, min_periods=1).mean()
            feature_dict[f"rollstd{W}"] = df["target"].shift(1).rolling(W, min_periods=1).std()

        # Optional cross-topic covariates (current + short lags) and simple interaction
        if extra_series:
            for cov_name, cov_vals in extra_series.items():
                cov_trimmed = list(cov_vals[-required:])
                if len(cov_trimmed) < required:
                    cov_trimmed = [np.nan] * (required - len(cov_trimmed)) + cov_trimmed
                cov_series = pd.Series(cov_trimmed, index=idx)
                feature_dict[f"{cov_name}_cur"] = cov_series
                for L in (1, 3):
                    feature_dict[f"{cov_name}_lag{L}"] = cov_series.shift(L)
            if "t_cur" in feature_dict and "rh_cur" in feature_dict:
                feature_dict["t_x_rh"] = feature_dict["t_cur"] * feature_dict["rh_cur"]

        feature_dict.update(
            {
                "hour": idx.hour,
                "dow": idx.dayofweek,
                "month": idx.month,
                "is_weekend": (idx.dayofweek >= 5).astype(int),
            }
        )

        X = pd.DataFrame(feature_dict, index=idx)
        return X.iloc[-1:]

    def _extract_features(self, target_topic):
        with self.buffer_lock:
            values = list(self.data_buffer.get(target_topic, []))
            extra = {}
            for cov_topic in ("t", "rh"):
                if cov_topic in self.data_buffer:
                    extra[cov_topic] = list(self.data_buffer[cov_topic])
        return self._extract_features_from_series(values, extra_series=extra)

    def _ensure_model(self, target_name, n_features):
        if target_name not in self.models:
            # Conservative defaults to avoid exploding updates
            self.models[target_name] = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=1e-4,
                learning_rate="optimal",
                eta0=0.05,
                power_t=0.5,
                random_state=42,
                max_iter=1,
                warm_start=True,
            )
            self.model_initialized[target_name] = False
        return self.models[target_name]

    def _get_scaler(self, target_name):
        if target_name not in self.scalers:
            self.scalers[target_name] = StandardScaler()
        return self.scalers[target_name]

    def _detect_anomaly(self, topic, value):
        buf = list(self.data_buffer.get(topic, []))
        if not buf:
            return None
        recent = buf[-self.short_window :]
        if len(recent) < 2:
            return None
        arr = np.array([v for v in recent if not np.isnan(v)])
        if arr.size < 2:
            return None
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        z_score = (value - mean) / std if std > 0 else 0.0
        q_high = self.quantiles[topic].quantile(self.quantile_q)
        q_low = self.quantiles[topic].quantile(1 - self.quantile_q)
        high = q_high is not None and value > q_high
        low = q_low is not None and value < q_low
        flagged = abs(z_score) >= self.z_threshold or high or low
        if flagged and self.verb:
            print(f"OnlineForecaster: anomaly {topic} val={value} z={z_score:.2f} q_high={q_high} q_low={q_low}")
        return {
            "z_score": z_score,
            "recent_mean": mean,
            "recent_std": std,
            "q_high": q_high,
            "q_low": q_low,
            "flagged": flagged,
        }

    def _recent_stats(self, topic):
        buf = [v for v in self.data_buffer.get(topic, []) if not np.isnan(v)]
        if not buf:
            return {}
        arr = np.array(buf[-self.short_window :])
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": arr.size,
        }

    def _clamp_prediction(self, topic, raw_pred):
        stats = self._recent_stats(topic) or {}
        if not stats or not np.isfinite(raw_pred):
            return raw_pred
        mean = stats.get("mean", 0.0)
        std = stats.get("std", 0.0)
        mn = stats.get("min", mean)
        mx = stats.get("max", mean)
        # Looser band to follow spikes: ±3σ and allow growth beyond recent max
        lo = mean - 3 * std if std > 0 else mn
        hi = mean + 3 * std if std > 0 else mx * 1.3
        lo = min(lo, mn)
        hi = max(hi, mx * 1.3 if mx else hi)
        return float(np.clip(raw_pred, lo, hi))

    def _fallback_prediction(self, topic):
        """Use a simple mean of the last 5 valid readings as a safe fallback."""
        buf = [v for v in self.data_buffer.get(topic, []) if not np.isnan(v)]
        if not buf:
            return None
        recent = buf[-5:] if len(buf) >= 5 else buf
        return float(np.mean(recent)) if recent else None

    def _train_and_predict(self, topic, value, timestamp):
        target_name = self.TOPIC_TO_COLUMN[topic]
        features = self._extract_features(topic)
        if features is None:
            return None
        # Replace NaNs to keep SGDRegressor happy; use column means then zeros as fallback.
        features_filled = features.fillna(features.mean(numeric_only=True)).fillna(0.0)

        model = self._ensure_model(target_name, features.shape[1])
        scaler = self._get_scaler(target_name)

        pending = self.pending_feature.get(target_name)
        if pending is not None and not np.isnan(value):
            pending_filled = pending.fillna(pending.mean(numeric_only=True)).fillna(0.0)
            scaler.partial_fit(pending_filled.values)
            X_train = scaler.transform(pending_filled.values)
            model.partial_fit(X_train, np.array([value]))
            self.model_initialized[target_name] = True
            self.train_counts[target_name] = self.train_counts.get(target_name, 0) + 1

        prediction = None
        if self.model_initialized.get(target_name, False) and self.train_counts.get(target_name, 0) >= self.min_train_samples:
            try:
                scaler.partial_fit(features_filled.values)
                X_pred = scaler.transform(features_filled.values)
                raw_pred = float(model.predict(X_pred)[0])
                prediction = self._clamp_prediction(topic, raw_pred)
            except NotFittedError:
                prediction = None
        else:
            # Fallback to persistence/rolling mean until the model has seen enough samples
            stats = self._recent_stats(topic) or {}
            last_vals = [v for v in self.data_buffer.get(topic, []) if not np.isnan(v)]
            last_val = last_vals[-1] if last_vals else np.nan
            if not np.isnan(last_val):
                prediction = float(last_val)
            elif stats:
                prediction = float(stats.get("mean", np.nan))

        # If model pred exists but deviates too much from recent band, fall back to mean-of-last-5
        fallback_mean5 = self._fallback_prediction(topic)
        if prediction is not None and fallback_mean5 is not None:
            stats = self._recent_stats(topic) or {}
            mean = stats.get("mean", fallback_mean5)
            std = stats.get("std", 0.0)
            threshold = 2 * std if std > 0 else abs(mean) * 0.1
            if threshold is None or not np.isfinite(threshold):
                threshold = 0
            if abs(prediction - mean) > threshold:
                prediction = fallback_mean5

        # Spike bypass: if current jump is large vs last value, favor persistence
        last_vals = [v for v in self.data_buffer.get(topic, []) if not np.isnan(v)]
        if last_vals:
            last_val = last_vals[-1]
            jump = abs(value - last_val)
            stats = self._recent_stats(topic) or {}
            std = stats.get("std", 0.0)
            threshold = 2 * std if std > 0 else abs(last_val) * 0.1
            if np.isfinite(jump) and np.isfinite(threshold) and jump > threshold:
                prediction = float(last_val)

        # Keep the latest feature row for the next actual value
        self.pending_feature[target_name] = features_filled

        if prediction is None:
            prediction = self._fallback_prediction(topic)

        if prediction is None:
            return None

        preds = {"online_pred": prediction}
        self.last_predictions[target_name] = preds
        if self.dbWriter is not None:
            # Write single-step online forecast alongside observations
            self.dbWriter.write_data(
                "environment",
                {"topic": topic},
                preds,
                timestamp or datetime.now().isoformat(),
            )
        return preds

    def _handle_topic(self, topic, data):
        if not data:
            return
        latest = data[-1]
        value = latest.get("value")
        ts = latest.get("key")
        processed = self._update_buffer(topic, value)
        anomaly = self._detect_anomaly(topic, processed)
        preds = self._train_and_predict(topic, processed, ts)
        if preds and anomaly and anomaly.get("flagged") and self.verb:
            print(f"OnlineForecaster: {topic} anomaly flagged with prediction {preds}")

    # --- required listener callbacks --------------------------------------
    def on_new_window_pt08_s1_co(self, data):
        self._handle_topic("pt08_s1_co", data)

    def on_new_window_pt08_s2_nmhc(self, data):
        self._handle_topic("pt08_s2_nmhc", data)

    def on_new_window_pt08_s3_nox(self, data):
        self._handle_topic("pt08_s3_nox", data)

    def on_new_window_pt08_s4_no2(self, data):
        self._handle_topic("pt08_s4_no2", data)

    def on_new_window_pt08_s5_o3(self, data):
        self._handle_topic("pt08_s5_o3", data)

    def on_new_window_t(self, data):
        self._handle_topic("t", data)

    def on_new_window_rh(self, data):
        self._handle_topic("rh", data)

    def on_new_window_ah(self, data):
        self._handle_topic("ah", data)
