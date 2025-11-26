import json
import os
from collections import deque
from datetime import datetime
from threading import Lock, Thread, Event

import joblib
import numpy as np
import pandas as pd
import traceback
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
        "pt08_s1_co": "PT08.S1(CO)",
        "pt08_s2_nmhc": "PT08.S2(NMHC)",
        "pt08_s3_nox": "PT08.S3(NOx)",
        "pt08_s4_no2": "PT08.S4(NO2)",
        "pt08_s5_o3": "PT08.S5(O3)",
        "t": "T",
        "rh": "RH",
        "ah": "AH",
    }

    # reverse map: column name -> topic key
    COLUMN_TO_TOPIC = {v: k for k, v in TOPIC_TO_COLUMN.items()}

    def __new__(cls, artifacts_dir=None, target=None, horizons=None, verb=False):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, artifacts_dir=None, target=None, horizons=None, verb=False, lag_hours=None, roll_windows=None, required_samples=None, max_buffer_size=50):
        """
        required_samples: int or dict mapping target_name -> int. If int, applies to all targets.
        If not provided, defaults to max(lag_hours)+1 behavior.
        max_buffer_size: maximum deque length for each topic buffer.
        """
        already_init = getattr(self, '_initialized', False)
        # allow updating configuration after first init without re-creating background thread
        if already_init:
            # update config values if provided
            if verb is not None:
                self.verb = verb
            if target is not None:
                self.target = target
            if horizons is not None:
                self.horizons = horizons
            if lag_hours is not None:
                self.lag_hours = tuple(lag_hours)
            if roll_windows is not None:
                self.roll_windows = tuple(roll_windows)
            if required_samples is not None:
                self.required_samples = required_samples
            if max_buffer_size is not None:
                self.max_buffer_size = max_buffer_size
            return

        super().__init__()

        self.verb = verb
        # do NOT hard-code a default target; allow None and resolve later
        self.target = target
        self.horizons = horizons if horizons is not None else [1, 2, 3]

        # feature configuration: lags and rolling window sizes
        # default mirrors previous hard-coded behavior
        self.lag_hours = tuple(lag_hours) if lag_hours is not None else (1, 2, 3, 6, 12, 24)
        self.roll_windows = tuple(roll_windows) if roll_windows is not None else (3, 6, 12)

        # required_samples: exact number of historical samples to use for feature creation
        # can be an int (applies to all targets) or a dict mapping target_name -> int
        self.required_samples = required_samples

        if artifacts_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level from offline_forcasting/ to app/, then join with artifacts
            self.artifacts_dir = os.path.join(os.path.dirname(current_dir), 'artifacts')
        else:
            self.artifacts_dir = artifacts_dir

        # buffers
        self.data_buffer = {}  # topic -> deque
        self.buffer_lock = Lock()
        self.max_buffer_size = max_buffer_size

        # models & report (load all available targets)
        # structure: self.models[target_str][H{h}] = model
        self.models = {}
        self.reports = {}
        self._load_models()
        self._load_reports()

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
        # scan artifacts dir for model files and load them grouped by target
        try:
            files = os.listdir(self.artifacts_dir)
        except Exception:
            files = []

        # Workaround: some saved models contain numpy Generator state that
        # references PCG64 (numpy.random._pcg64.PCG64). Older/newer numpy
        # runtimes may not recognize that BitGenerator name when unpickling.
        # Register PCG64 in numpy's pickle constructors if possible so
        # joblib.load can reconstruct RNG state embedded in model objects.
        try:
            try:
                # local imports to avoid hard dependency at module import time
                from numpy.random._pcg64 import PCG64
                import numpy.random._pickle as _npr_pickle
                # key expected by numpy's unpickler is the class name 'PCG64'
                _npr_pickle._bit_generator_constructors['PCG64'] = PCG64
            except Exception:
                # if private API differs or import fails, skip silently
                pass
        except Exception:
            pass

        if self.verb:
            try:
                print(f"Loading models from artifacts_dir={self.artifacts_dir}, total_files={len(files)}")
            except Exception:
                pass

        import re
        pattern = re.compile(r"^model_(.+)_H(\d+)_hgb\.joblib$")
        for fn in files:
            m = pattern.match(fn)
            if not m:
                continue
            target_name = m.group(1)  # as in filename (parentheses etc retained)
            h = int(m.group(2))
            model_path = os.path.join(self.artifacts_dir, fn)
            try:
                mdl = joblib.load(model_path)
                self.models.setdefault(target_name, {})[f"H{h}"] = mdl
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
                if self.verb:
                    import traceback as _tb
                    _tb.print_exc()

            # normalize model keys: if filenames used '-' instead of '/', allow both
            # (no change to storage keys — they match filenames/column names)

    def _load_reports(self):
        # load any report_{target}.json files into self.reports[target]
        try:
            files = os.listdir(self.artifacts_dir)
        except Exception:
            files = []

        import re
        pattern = re.compile(r"^report_(.+)\.json$")
        for fn in files:
            m = pattern.match(fn)
            if not m:
                continue
            target_name = m.group(1)
            report_path = os.path.join(self.artifacts_dir, fn)
            try:
                with open(report_path, 'r') as f:
                    self.reports[target_name] = json.load(f)
                if self.verb:
                    print(f"Loaded report for {target_name} from {report_path}")
            except Exception as e:
                if self.verb:
                    print(f"Error loading report {report_path}: {e}")

    def _update_buffer(self, topic, value):
        """
        Update buffer with latest scalar reading for `topic`. If buffers have
        sufficient aligned history, call predict() (outside the lock).
        """
        # update buffer with the new sample (no auto-prediction here)
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
            # Note: we no longer auto-predict here. Handlers (on_new_window_*)
            # call predict() explicitly after updating the buffer. The periodic
            # prediction loop will also run predictions for loaded targets.

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

    def _get_required_samples_for_target(self, target_name):
        """Return exact required samples (int) for a target_name.
        If `self.required_samples` is an int, return that.
        If it's a dict, return value for target_name if present.
        Otherwise default to max(self.lag_hours) + 1.
        """
        if self.required_samples is None:
            return max(self.lag_hours) + 1
        if isinstance(self.required_samples, int):
            return int(self.required_samples)
        if isinstance(self.required_samples, dict):
            return int(self.required_samples.get(target_name, max(self.lag_hours) + 1))
        # fallback
        return max(self.lag_hours) + 1

    def _extract_features(self, lag_hours=None, roll_windows=None, target=None):
        # target: target column name like 'NO2(GT)' or similar. If not provided, use self.target
        # lag_hours / roll_windows: optional overrides (tuples). If not provided, use instance config.
        target_name = target if target is not None else self.target
        # if still None, try to pick the first loaded model target as default
        if target_name is None:
            target_name = next(iter(self.models.keys()), None)
            if target_name is None:
                if self.verb:
                    print('No target specified and no models loaded')
                return None
        target_topic = self.COLUMN_TO_TOPIC.get(target_name)
        lag_hours = tuple(lag_hours) if lag_hours is not None else self.lag_hours
        roll_windows = tuple(roll_windows) if roll_windows is not None else self.roll_windows
        with self.buffer_lock:
            required = self._get_required_samples_for_target(target_name)
            # require that target's own buffer has at least `required` samples
            if target_topic not in self.data_buffer or len(self.data_buffer[target_topic]) < required:
                if self.verb:
                    have = len(self.data_buffer.get(target_topic, []))
                    print(f'Insufficient buffer for {target_topic or target_name}: {have}, need at least {required}')
                return None

            data = {}
            for topic, col in self.TOPIC_TO_COLUMN.items():
                vals = list(self.data_buffer.get(topic, []))
                if len(vals) >= required:
                    values = vals[-required:]
                else:
                    # left-pad with NaNs so all columns have length `required`
                    pad = [np.nan] * (required - len(vals))
                    values = pad + vals
                data[col] = values

            # Create DataFrame with DatetimeIndex (hourly frequency going backwards from now)
            now = datetime.now()
            timestamps = pd.date_range(end=now, periods=required, freq='h')
            df = pd.DataFrame(data, index=timestamps)
            target_col = self.TOPIC_TO_COLUMN.get(target_topic, target_name)
            if target_col not in df.columns:
                if self.verb:
                    print(f"Column {target_col} not found in data")
                return None

            # Add time features
            df_time = self._add_time_features(required)
            df = pd.concat([df, df_time], axis=1)

            # Extract numeric columns (excluding time features)
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ['hour', 'dow', 'month', 'is_weekend']]
            
            # Collect all features in a dictionary to avoid fragmentation
            feature_dict = {}
            
            # Create lag and rolling features (matching training notebook)
            for c in num_cols:
                for L in lag_hours:
                    feature_dict[f"{c}_lag{L}"] = df[c].shift(L)
                for W in roll_windows:
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

    def predict(self, horizon=None, target=None):
        # target: string matching loaded model keys (e.g. 'NO2(GT)', 'CO(GT)')
        target_key = target if target is not None else self.target
        if target_key is None:
            target_key = next(iter(self.models.keys()), None)
            if target_key is None:
                if self.verb:
                    print('No target specified and no models loaded to predict')
                return None
        features = self._extract_features(target=target_key)
        if features is None or len(features) == 0:
            if self.verb:
                print('Insufficient data for prediction')
            # keep previous predictions untouched
            return None
        preds = {}
        horizons_to_predict = [horizon] if horizon is not None else self.horizons

        models_for_target = self.models.get(target_key, {})

        for h in horizons_to_predict:
            key = f'H{h}'
            if key in models_for_target:
                model = models_for_target[key]
                try:
                    # Handle sklearn Pipeline objects - feature_names_in_ is on the final estimator
                    fn = None
                    if hasattr(model, 'feature_names_in_'):
                        fn = model.feature_names_in_
                    elif hasattr(model, 'named_steps'):
                        # attempt to find a step that has feature_names_in_
                        for step in model.named_steps.values():
                            if hasattr(step, 'feature_names_in_'):
                                fn = step.feature_names_in_
                                break

                    if fn is not None:
                        Xord = features.reindex(columns=fn, fill_value=np.nan)
                    else:
                        Xord = features

                    p = model.predict(Xord)[0]
                    preds[f'H+{h}'] = float(p)
                except Exception as e:
                    if self.verb:
                        print(f'Error predicting {target_key} H+{h}: {e}')
                        traceback.print_exc()
                    preds[f'H+{h}'] = None
            else:
                preds[f'H+{h}'] = None

        # store per-target last predictions
        if not hasattr(self, 'last_predictions') or self.last_predictions is None:
            self.last_predictions = {}
        self.last_predictions[target_key] = preds
        return preds

    def _prediction_loop(self):
        while not self._stop_event.is_set():
            try:
                # run predictions for all loaded targets
                for target_key in list(self.models.keys()):
                    try:
                        preds = self.predict(target=target_key)
                        if preds:
                            print(f'OfflineForecaster: [Periodic] {target_key} -> {preds}')
                    except Exception:
                        if self.verb:
                            print(f'Error predicting {target_key} in periodic loop')
            except Exception:
                if self.verb:
                    print('Error in prediction loop')
            self._stop_event.wait(self.prediction_interval)

        def inspect_models(self):
            """Return summary of loaded models and any detected feature_names_in_.

            Useful for debugging whether models were loaded and what feature names
            they expect. Does not run predictions.
            """
            out = {}
            for target, models in self.models.items():
                out[target] = {}
                for k, m in models.items():
                    fn = None
                    try:
                        if hasattr(m, 'feature_names_in_'):
                            fn = list(m.feature_names_in_)
                        elif hasattr(m, 'named_steps'):
                            for step in m.named_steps.values():
                                if hasattr(step, 'feature_names_in_'):
                                    fn = list(step.feature_names_in_)
                                    break
                    except Exception:
                        fn = None
                    out[target][k] = {'feature_names_in_': fn}
            if self.verb:
                try:
                    print('inspect_models:', out)
                except Exception:
                    pass
            return out

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

    def on_new_window_pt08_s1_co(self, data):
        if not data:
            return
        latest = data[-1]
        try:
            self._update_buffer('pt08_s1_co', latest['value'])
        except Exception:
            pass
        target_name = self.TOPIC_TO_COLUMN['pt08_s1_co']
        preds = self.predict(target=target_name)
        if preds:
            print(f"OfflineForecaster: {target_name} -> {preds}")

    def on_new_window_pt08_s2_nmhc(self, data):
        if not data:
            return
        latest = data[-1]
        try:
            self._update_buffer('pt08_s2_nmhc', latest['value'])
        except Exception:
            pass
        target_name = self.TOPIC_TO_COLUMN['pt08_s2_nmhc']
        preds = self.predict(target=target_name)
        if preds:
            print(f"OfflineForecaster: {target_name} -> {preds}")

    def on_new_window_pt08_s3_nox(self, data):
        if not data:
            return
        latest = data[-1]
        try:
            self._update_buffer('pt08_s3_nox', latest['value'])
        except Exception:
            pass
        target_name = self.TOPIC_TO_COLUMN['pt08_s3_nox']
        preds = self.predict(target=target_name)
        if preds:
            print(f"OfflineForecaster: {target_name} -> {preds}")

    def on_new_window_pt08_s4_no2(self, data):
        if not data:
            return
        latest = data[-1]
        try:
            self._update_buffer('pt08_s4_no2', latest['value'])
        except Exception:
            pass
        target_name = self.TOPIC_TO_COLUMN['pt08_s4_no2']
        preds = self.predict(target=target_name)
        if preds:
            print(f"OfflineForecaster: {target_name} -> {preds}")

    def on_new_window_pt08_s5_o3(self, data):
        if not data:
            return
        latest = data[-1]
        try:
            self._update_buffer('pt08_s5_o3', latest['value'])
        except Exception:
            pass
        target_name = self.TOPIC_TO_COLUMN['pt08_s5_o3']
        preds = self.predict(target=target_name)
        if preds:
            print(f"OfflineForecaster: {target_name} -> {preds}")

    def on_new_window_t(self, data):
        if not data:
            return
        latest = data[-1]
        try:
            self._update_buffer('t', latest['value'])
        except Exception:
            pass
        target_name = self.TOPIC_TO_COLUMN['t']
        preds = self.predict(target=target_name)
        if preds:
            print(f"OfflineForecaster: {target_name} -> {preds}")

    def on_new_window_rh(self, data):
        if not data:
            return
        latest = data[-1]
        try:
            self._update_buffer('rh', latest['value'])
        except Exception:
            pass
        target_name = self.TOPIC_TO_COLUMN['rh']
        preds = self.predict(target=target_name)
        if preds:
            print(f"OfflineForecaster: {target_name} -> {preds}")

    def on_new_window_ah(self, data):
        if not data:
            return
        latest = data[-1]
        try:
            self._update_buffer('ah', latest['value'])
        except Exception:
            pass
        target_name = self.TOPIC_TO_COLUMN['ah']
        preds = self.predict(target=target_name)
        if preds:
            print(f"OfflineForecaster: {target_name} -> {preds}")
