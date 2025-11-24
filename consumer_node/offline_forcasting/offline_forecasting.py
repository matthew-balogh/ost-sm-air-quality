import json
import os
from collections import deque
from datetime import datetime
from threading import Lock, Thread, Event

import joblib
import numpy as np
import pandas as pd
from listeners.sliding_window_listener import SlidingWindowListener


class OfflineForecaster(SlidingWindowListener):
    """
    Singleton listener that buffers latest sensor values and makes predictions
    for NO2 using pre-trained models. When the buffer has enough history to
    compute features (lags + rolling windows) the code will automatically
    call predict() after each buffer update. A background thread also calls
    predict() periodically (default every 10 seconds).
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

    def __new__(cls, *args, **kwargs):
        # Accept arbitrary constructor args/kwargs so __new__ doesn't raise
        # when new parameters (like lag_hours) are added to __init__.
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, artifacts_dir=None, target="NO2(GT)", horizons=None, verb=False,
                 lag_hours=None, roll_windows=None,
                 write_predictions=False, influx_table='predictions', influx_tags=None,
                 influx_verbose=False, predict_all_topics=False):
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

        # feature configuration (make configurable so users can reduce history requirement)
        self.lag_hours = tuple(lag_hours) if lag_hours is not None else (1, 2, 3, 6, 12, 24)
        self.roll_windows = tuple(roll_windows) if roll_windows is not None else (3, 6, 12)

        # InfluxDB write options for predictions
        self.write_predictions = write_predictions
        self.influx_table = influx_table
        self.influx_tags = influx_tags or {"topic": "no2_forecast"}
        self.influx_verbose = influx_verbose
        self._influx_writer = None

        # If True, predict for all topics (uses predict_all()); otherwise
        # legacy behaviour targets a single configured target (NO2 by default)
        self.predict_all_topics = bool(predict_all_topics)

        # models & report
        # models: legacy single-target mapping kept for compatibility
        self.models = {}
        # models_by_topic: map topic_key -> { 'H1': model, ... }
        self.models_by_topic = {}
        self.report = None
        self._load_models()
        self._load_report()

        # prediction control
        self.prediction_interval = 10
        self.last_predictions = None
        self._stop_event = Event()
        self._prediction_thread = Thread(target=self._prediction_loop, daemon=True)
        self._prediction_thread.start()

        self._initialized = True
        if self.verb:
            print('OfflineForecaster initialized; models:', list(self.models.keys()))

    def _load_models(self):
        # Load models for all known topics (if available). Models are expected
        # to be named like `model_<target_safe>_H<h>_hgb.joblib` where
        # <target_safe> corresponds to the column name in TOPIC_TO_COLUMN.
        for topic_key, col_name in self.TOPIC_TO_COLUMN.items():
            target_safe = col_name.replace('/', '-')
            loaded = {}
            for h in self.horizons:
                model_path = os.path.join(self.artifacts_dir, f"model_{target_safe}_H{h}_hgb.joblib")
                if os.path.exists(model_path):
                    try:
                        loaded[f"H{h}"] = joblib.load(model_path)
                        if self.verb:
                            print(f"Loaded model for {topic_key} H+{h} from {model_path}")
                    except Exception as e:
                        if self.verb:
                            print(f"Error loading model {model_path}: {e}")
                else:
                    if self.verb:
                        print(f"Model not found: {model_path}")
            if loaded:
                self.models_by_topic[topic_key] = loaded
                # record feature names if available for debugging
                info = {}
                for k, m in loaded.items():
                    try:
                        fn = getattr(m, 'feature_names_in_', None)
                        if fn is None and hasattr(m, 'named_steps') and hasattr(m.named_steps.get('model'), 'feature_names_in_'):
                            fn = m.named_steps['model'].feature_names_in_
                        info[k] = {'feature_names': list(fn) if fn is not None else None}
                    except Exception:
                        info[k] = {'feature_names': None}
                setattr(self, 'models_info_' + topic_key, info)

        # keep legacy single-target mapping for backward compatibility
        # determine target_topic from TOPIC_TO_COLUMN mapping
        self.target_topic = None
        for k, v in self.TOPIC_TO_COLUMN.items():
            if v == self.target:
                self.target_topic = k
                break
        if self.target_topic and self.target_topic in self.models_by_topic:
            self.models = self.models_by_topic[self.target_topic]

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
        required_min = max(self.lag_hours) + 1

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

    def _add_time_features(self, n_samples):
        """
        Create time features for n_samples, using current time as the latest timestamp.
        Assumes hourly frequency going backwards.
        """
        now = datetime.now()
        # Create DatetimeIndex going backwards from now (hourly frequency)
        timestamps = pd.date_range(end=now, periods=n_samples, freq='h')
        return pd.DataFrame({
            'hour': timestamps.hour,
            'dow': timestamps.dayofweek,
            'month': timestamps.month,
            'is_weekend': (timestamps.dayofweek >= 5).astype(int),
        }, index=timestamps)

    def _extract_features(self):
        with self.buffer_lock:
            if 'no2_gt' not in self.data_buffer:
                if self.verb:
                    print('no2_gt is missing from buffer')
                return None
            buffer_lengths = [len(v) for v in self.data_buffer.values()]
            if not buffer_lengths or min(buffer_lengths) < max(self.lag_hours) + 1:
                if self.verb:
                    print(f'Insufficient buffer length: {min(buffer_lengths) if buffer_lengths else 0}, need at least {max(self.lag_hours) + 1}')
                return None
            min_length = min(buffer_lengths)

            data = {}
            for topic, col in self.TOPIC_TO_COLUMN.items():
                if topic in self.data_buffer:
                    values = list(self.data_buffer[topic])[-min_length:]
                    data[col] = values
                else:
                    data[col] = [np.nan] * min_length

            # Create DataFrame with DatetimeIndex (hourly frequency going backwards from now)
            now = datetime.now()
            timestamps = pd.date_range(end=now, periods=min_length, freq='h')
            df = pd.DataFrame(data, index=timestamps)

            target_col = self.TOPIC_TO_COLUMN.get('no2_gt', self.target)
            if target_col not in df.columns:
                if self.verb:
                    print(f"Column {target_col} not found in data")
                return None

            # Add time features
            df_time = self._add_time_features(min_length)
            df = pd.concat([df, df_time], axis=1)

            # Extract numeric columns (excluding time features)
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ['hour', 'dow', 'month', 'is_weekend']]
            
            # Collect all features in a dictionary to avoid fragmentation
            feature_dict = {}
            
            # Create lag and rolling features (matching training notebook)
            for c in num_cols:
                for L in self.lag_hours:
                    feature_dict[f"{c}_lag{L}"] = df[c].shift(L)
                for W in self.roll_windows:
                    feature_dict[f"{c}_rollmean{W}"] = df[c].shift(1).rolling(W, min_periods=1).mean()
                    feature_dict[f"{c}_rollstd{W}"] = df[c].shift(1).rolling(W, min_periods=1).std()

            # Add time features
            feature_dict['hour'] = df_time['hour']
            feature_dict['dow'] = df_time['dow']
            feature_dict['month'] = df_time['month']
            feature_dict['is_weekend'] = df_time['is_weekend']
            
            # Create DataFrame all at once to avoid fragmentation
            X = pd.DataFrame(feature_dict, index=df.index)

            if len(X) > 0:
                # Return the latest row (most recent features)
                return X.iloc[-1:].copy()

            if self.verb:
                print('No data after feature extraction')
            return None

    def predict(self, horizon=None):
        # If configured, predict for all topics and return a dict of topic->preds
        if getattr(self, 'predict_all_topics', False):
            return self.predict_all()

        # default predict for configured target_topic (legacy behaviour)
        topic = getattr(self, 'target_topic', None)
        return self.predict_for_topic(topic, horizon=horizon)

    def predict_for_topic(self, topic_key, horizon=None):
        """
        Predict for a specific topic_key (e.g., 'no2_gt'). Returns dict of
        horizon predictions like {'H+1': value, ...} or None if not possible.
        """
        if topic_key is None:
            if self.verb:
                print('No target topic configured for prediction')
            return None

        if topic_key not in self.models_by_topic:
            # No trained models for this topic: fall back to a simple persistence forecast
            # (repeat last observed value) so we can still produce outputs for all topics.
            if self.verb:
                print(f'No trained models for topic {topic_key}; using persistence fallback')
            with self.buffer_lock:
                buf = self.data_buffer.get(topic_key, None)
                if not buf:
                    if self.verb:
                        print(f'No buffer data for {topic_key}; cannot fallback')
                    return None
                # find last non-nan value
                last_val = None
                for v in reversed(buf):
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        last_val = v
                        break
                if last_val is None:
                    if self.verb:
                        print(f'Buffer contains only NaN for {topic_key}; cannot fallback')
                    return None

            preds = {}
            horizons_to_predict = [horizon] if horizon is not None else self.horizons
            for h in horizons_to_predict:
                preds[f'H+{h}'] = float(last_val)

            # write fallback predictions as well so downstream can consume them
            self.last_predictions = preds
            if self.write_predictions:
                try:
                    if self._influx_writer is None:
                        # attempt to import writer as done elsewhere
                        try:
                            pkg = __import__('InfluxDB.InfluxDbUtilities', fromlist=['DatabaseWriter'])
                            DatabaseWriter = getattr(pkg, 'DatabaseWriter')
                            self._influx_writer = DatabaseWriter(verbose=self.influx_verbose)
                        except Exception:
                            self._influx_writer = None
                    if self._influx_writer is not None:
                        timestamp = datetime.now().strftime("%d/%m/%Y %H.%M.%S")
                        try:
                            self._influx_writer.write_prediction(topic_key, preds, timestamp)
                        except Exception as e:
                            if self.verb:
                                print(f'Error writing fallback predictions for {topic_key}: {e}')
                except Exception:
                    if self.verb:
                        print('Unexpected error while attempting to write fallback predictions')

            return preds

        features = self._extract_features()
        if features is None or len(features) == 0:
            if self.verb:
                print('Insufficient data for prediction')
            return None

        preds = {}
        horizons_to_predict = [horizon] if horizon is not None else self.horizons
        models = self.models_by_topic.get(topic_key, {})
        for h in horizons_to_predict:
            key = f'H{h}'
            if key in models:
                model = models[key]
                try:
                    if hasattr(model, 'feature_names_in_'):
                        fn = model.feature_names_in_
                        Xord = features.reindex(columns=fn, fill_value=np.nan)
                    elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('model'), 'feature_names_in_'):
                        fn = model.named_steps['model'].feature_names_in_
                        Xord = features.reindex(columns=fn, fill_value=np.nan)
                    else:
                        Xord = features
                    p = model.predict(Xord)[0]
                    preds[f'H+{h}'] = float(p)
                except Exception as e:
                    if self.verb:
                        print(f'Error predicting {topic_key} H+{h}: {e}')
                    preds[f'H+{h}'] = None
            else:
                preds[f'H+{h}'] = None

        self.last_predictions = preds

        # write per-topic prediction back to Influx via DatabaseWriter helper
        if self.write_predictions:
            try:
                if self._influx_writer is None:
                    # Try multiple import paths
                    tried = []
                    loaded = False
                    for module_path in (
                        'InfluxDB.InfluxDbUtilities',
                        'consumer_node.InfluxDB.InfluxDbUtilities',
                        '..InfluxDB.InfluxDbUtilities',
                    ):
                        try:
                            pkg = __import__(module_path, fromlist=['DatabaseWriter'])
                            DatabaseWriter = getattr(pkg, 'DatabaseWriter')
                            self._influx_writer = DatabaseWriter(verbose=self.influx_verbose)
                            loaded = True
                            break
                        except Exception as e:
                            tried.append((module_path, str(e)))
                    if not loaded:
                        if self.verb:
                            print('Could not import DatabaseWriter from any known path:')
                            for p, e in tried:
                                print(f'  {p}: {e}')
                        self._influx_writer = None

                if self._influx_writer is not None:
                    timestamp = datetime.now().strftime("%d/%m/%Y %H.%M.%S")
                    try:
                        # use DatabaseWriter.write_prediction to write prefixed fields
                        self._influx_writer.write_prediction(topic_key, preds, timestamp)
                    except Exception as e:
                        if self.verb:
                            print(f'Error writing predictions to InfluxDB: {e}')
            except Exception:
                if self.verb:
                    print('Unexpected error while attempting to write predictions')

        return preds

    def predict_all(self):
        """
        Predict for all known topics (based on TOPIC_TO_COLUMN). Returns a
        dictionary topic_key -> preds or None when prediction not possible.
        This will call `predict_for_topic` for each topic so fallback
        persistence predictions are also produced when trained models are
        missing.
        """
        results = {}
        for topic_key in list(self.TOPIC_TO_COLUMN.keys()):
            try:
                p = self.predict_for_topic(topic_key)
                results[topic_key] = p
            except Exception as e:
                if self.verb:
                    print(f'Error in predict_all for {topic_key}: {e}')
                results[topic_key] = None
        return results

    def list_models(self):
        """Return a dict of topics -> available horizons (models) or empty list."""
        res = {}
        for topic in self.TOPIC_TO_COLUMN.keys():
            res[topic] = list(self.models_by_topic.get(topic, {}).keys())
        return res

    def run_diagnostics(self):
        """Run a one-shot diagnostic: list models and attempt one prediction per topic."""
        info = {
            'models': self.list_models(),
            'predictions': {}
        }
        for topic in self.TOPIC_TO_COLUMN.keys():
            try:
                p = self.predict_for_topic(topic)
                info['predictions'][topic] = p
            except Exception as e:
                info['predictions'][topic] = {'error': str(e)}
        if self.verb:
            print('Diagnostics:', json.dumps(info, indent=2))
        return info

    def _prediction_loop(self):
        while not self._stop_event.is_set():
            try:
                # Periodically predict for all topics and print summary
                all_preds = self.predict_all()
                if all_preds:
                    # Only print compact summary for non-verbose mode
                    if not self.verb:
                        print(f'[Periodic Prediction] { {k: v for k, v in all_preds.items() if v} }')
                    else:
                        print(f'[Periodic Prediction - all] {all_preds}')
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

        # if self.verb:
        #     print(f"[OfflineForecaster] no2_gt updated, buffer lengths: {[len(v) for v in self.data_buffer.values()]}")
        # else:
        #     print('no2_gt window updated')

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
