import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
from threading import Lock, Thread, Event
from collections import deque

from listeners.sliding_window_listener import SlidingWindowListener


class OfflineForecaster(SlidingWindowListener):
    """
    Singleton listener that buffers latest sensor values and makes predictions
    for NO2 using pre-trained models. When the buffer has enough history to
    compute features (lags + rolling windows) the code will automatically
    call predict() after each buffer update. A background thread also calls
    predict() periodically (default every 5 seconds).
    """

    _instance = None
    _lock = Lock()

    TOPIC_TO_COLUMN = {
        "co_gt": "CO(GT)",
        "pt08_s1_co": "PT08.S1(CO)",
        "nmhc_gt": "NMHC(GT)",
        "c6h6_gt": "C6H6(GT)",
        "pt08_s2_nmhc": "PT08.S2(NMHC)",
        "nox_gt": "NOx(GT)",
        "pt08_s3_nox": "PT08.S3(NOx)",
        "no2_gt": "NO2(GT)",
        "pt08_s4_no2": "PT08.S4(NO2)",
        "pt08_s5_o3": "PT08.S5(O3)",
        "t": "T",
        "rh": "RH",
        "ah": "AH",
    }

    def __new__(cls, artifacts_dir=None, target="NO2(GT)", horizons=None, verb=False):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, artifacts_dir=None, target="NO2(GT)", horizons=None, verb=False):
        if getattr(self, '_initialized', False):
            return

        super().__init__()

        self.verb = verb
        self.target = target
        self.horizons = horizons if horizons is not None else [1, 2, 3]

        if artifacts_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'artifacts')
        else:
            self.artifacts_dir = artifacts_dir

        # buffers
        self.data_buffer = {}  # topic -> deque
        self.buffer_lock = Lock()
        self.max_buffer_size = 50

        # models & report
        self.models = {}
        self.report = None
        self._load_models()
        self._load_report()

        # prediction control
        self.prediction_interval = 5
        self.last_predictions = None
        self._stop_event = Event()
        self._prediction_thread = Thread(target=self._prediction_loop, daemon=True)
        self._prediction_thread.start()

        self._initialized = True
        if self.verb:
            print('OfflineForecaster initialized; models:', list(self.models.keys()))

    def _load_models(self):
        target_safe = self.target.replace('/', '-')
        for h in self.horizons:
            model_path = os.path.join(self.artifacts_dir, f"model_{target_safe}_H{h}_hgb.joblib")
            if os.path.exists(model_path):
                try:
                    self.models[f"H{h}"] = joblib.load(model_path)
                    if self.verb:
                        print(f"Loaded model H+{h} from {model_path}")
                except Exception as e:
                    print(f"Error loading model {model_path}: {e}")
            else:
                if self.verb:
                    print(f"Model not found: {model_path}")

    def _load_report(self):
        target_safe = self.target.replace('/', '-')
        report_path = os.path.join(self.artifacts_dir, f"report_{target_safe}.json")
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r') as f:
                    self.report = json.load(f)
                if self.verb:
                    print(f"Loaded report from {report_path}")
            except Exception as e:
                if self.verb:
                    print(f"Error loading report: {e}")

    def _update_buffer(self, topic, value):
        """
        Update buffer with latest scalar reading for `topic`. If buffers have
        sufficient aligned history, call predict() (outside the lock).
        """
        should_predict = False
        lag_hours = (1, 2, 3, 6, 12, 24)
        required_min = max(lag_hours) + 1

        with self.buffer_lock:
            if topic not in self.data_buffer:
                self.data_buffer[topic] = deque(maxlen=self.max_buffer_size)

            if value == -200 or value == "-200":
                processed_val = np.nan
            else:
                try:
                    processed_val = float(value)
                except (ValueError, TypeError):
                    processed_val = np.nan

            self.data_buffer[topic].append(processed_val)

            # readiness check: require target present and min length across buffers
            if 'no2_gt' in self.data_buffer:
                lengths = [len(v) for v in self.data_buffer.values()]
                if lengths and min(lengths) >= required_min:
                    should_predict = True

        if should_predict:
            try:
                preds = self.predict()
                if preds:
                    if self.verb:
                        print(f"[Auto-predict] {preds}")
                    else:
                        print(f"NO2(GT) Forecast: {preds}")
            except Exception as e:
                if self.verb:
                    print(f"Error running auto-predict: {e}")

    def _add_time_features(self, idx):
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.DatetimeIndex([datetime.now()])
        return pd.DataFrame({
            'hour': idx.hour,
            'dow': idx.dayofweek,
            'month': idx.month,
            'is_weekend': (idx.dayofweek >= 5).astype(int),
        }, index=idx)

    def _extract_features(self, lag_hours=(1, 2, 3, 6, 12, 24), roll_windows=(3, 6, 12)):
        print(self.data_buffer)
        with self.buffer_lock:
            if 'no2_gt' not in self.data_buffer:
                return None
            buffer_lengths = [len(v) for v in self.data_buffer.values()]
            if not buffer_lengths or min(buffer_lengths) < max(lag_hours) + 1:
                return None
            min_length = min(buffer_lengths)

            data = {}
            for topic, col in self.TOPIC_TO_COLUMN.items():
                if topic in self.data_buffer:
                    values = list(self.data_buffer[topic])[-min_length:]
                    data[col] = values
                else:
                    data[col] = [np.nan] * min_length

            now = datetime.now()
            timestamps = pd.date_range(end=now, periods=min_length, freq='H')
            df = pd.DataFrame(data, index=timestamps)

            target_col = self.TOPIC_TO_COLUMN.get('no2_gt', self.target)
            if target_col not in df.columns:
                return None

            df_time = self._add_time_features(df.index)
            df = pd.concat([df, df_time], axis=1)

            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ['hour', 'dow', 'month', 'is_weekend']]
            X = pd.DataFrame(index=df.index)

            for c in num_cols:
                for L in lag_hours:
                    X[f"{c}_lag{L}"] = df[c].shift(L)
                for W in roll_windows:
                    X[f"{c}_rollmean{W}"] = df[c].shift(1).rolling(W, min_periods=1).mean()
                    X[f"{c}_rollstd{W}"] = df[c].shift(1).rolling(W, min_periods=1).std()

            X[['hour', 'dow', 'month', 'is_weekend']] = df_time

            if len(X) > 0:
                return X.iloc[-1:].copy()
            return None

    def predict(self, horizon=None):
        features = self._extract_features()
        if features is None or len(features) == 0:
            if self.verb:
                print('Insufficient data for prediction')
            self.last_predictions = None
            return None

        preds = {}
        horizons_to_predict = [horizon] if horizon is not None else self.horizons
        for h in horizons_to_predict:
            key = f'H{h}'
            if key in self.models:
                model = self.models[key]
                try:
                    if hasattr(model, 'feature_names_in_'):
                        fn = model.feature_names_in_
                        Xord = features.reindex(columns=fn, fill_value=np.nan)
                    else:
                        Xord = features
                    p = model.predict(Xord)[0]
                    preds[f'H+{h}'] = float(p)
                except Exception as e:
                    if self.verb:
                        print(f'Error predicting H+{h}: {e}')
                    preds[f'H+{h}'] = None
            else:
                preds[f'H+{h}'] = None

        self.last_predictions = preds
        return preds

    def _prediction_loop(self):
        while not self._stop_event.is_set():
            try:
                preds = self.predict()
                if preds and not self.verb:
                    print(f'[Periodic Prediction] {preds}')
            except Exception:
                if self.verb:
                    print('Error in prediction loop')
            self._stop_event.wait(self.prediction_interval)

    def stop(self, join_timeout=2):
        self._stop_event.set()
        if hasattr(self, '_prediction_thread') and self._prediction_thread.is_alive():
            self._prediction_thread.join(timeout=join_timeout)

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass

    # sliding-window callbacks: update buffer with latest value
    def on_new_window_co_gt(self, data):
        if not data:
            return
        latest = data[-1]
        try:
            self._update_buffer('co_gt', latest['value'])
        except Exception:
            pass

    def on_new_window_pt08_s1_co(self, data):
        if not data:
            return
        latest = data[-1]
        try:
            self._update_buffer('pt08_s1_co', latest['value'])
        except Exception:
            pass

    def on_new_window_nmhc_gt(self, data):
        if not data:
            return
        latest = data[-1]
        try:
            self._update_buffer('nmhc_gt', latest['value'])
        except Exception:
            pass

    def on_new_window_c6h6_gt(self, data):
        if not data:
            return
        latest = data[-1]
        try:
            self._update_buffer('c6h6_gt', latest['value'])
        except Exception:
            pass

    def on_new_window_pt08_s2_nmhc(self, data):
        if not data:
            return
        latest = data[-1]
        try:
            self._update_buffer('pt08_s2_nmhc', latest['value'])
        except Exception:
            pass

    def on_new_window_nox_gt(self, data):
        if not data:
            return
        latest = data[-1]
        try:
            self._update_buffer('nox_gt', latest['value'])
        except Exception:
            pass

    def on_new_window_pt08_s3_nox(self, data):
        if not data:
            return
        latest = data[-1]
        try:
            self._update_buffer('pt08_s3_nox', latest['value'])
        except Exception:
            pass

    def on_new_window_no2_gt(self, data):
        if not data:
            return
        latest = data[-1]
        try:
            self._update_buffer('no2_gt', latest['value'])
        except Exception:
            pass

        if self.verb:
            print(f"[OfflineForecaster] no2_gt updated, buffer lengths: {[len(v) for v in self.data_buffer.values()]}")
        else:
            print('no2_gt window updated')

        preds = self.predict()
        if preds:
            if self.verb:
                print('Forecast for NO2(GT):', preds)
            else:
                print('NO2(GT) Forecast:', preds)

    def on_new_window_pt08_s4_no2(self, data):
        if not data:
            return
        latest = data[-1]
        try:
            self._update_buffer('pt08_s4_no2', latest['value'])
        except Exception:
            pass

    def on_new_window_pt08_s5_o3(self, data):
        if not data:
            return
        latest = data[-1]
        try:
            self._update_buffer('pt08_s5_o3', latest['value'])
        except Exception:
            pass

    def on_new_window_t(self, data):
        if not data:
            return
        latest = data[-1]
        try:
            self._update_buffer('t', latest['value'])
        except Exception:
            pass

    def on_new_window_rh(self, data):
        if not data:
            return
        latest = data[-1]
        try:
            self._update_buffer('rh', latest['value'])
        except Exception:
            pass

    def on_new_window_ah(self, data):
        if not data:
            return
        latest = data[-1]
        try:
            self._update_buffer('ah', latest['value'])
        except Exception:
            pass
