import os
import json
import joblib
import traceback
from collections import deque
from datetime import datetime
from threading import Lock

import numpy as np
import pandas as pd

from listeners.sliding_window_listener import SlidingWindowListener

try:
    from kafka import KafkaProducer
except Exception:
    KafkaProducer = None
import os
import json
import joblib
import traceback
from collections import deque
from datetime import datetime
from threading import Lock

import numpy as np
import pandas as pd

from listeners.sliding_window_listener import SlidingWindowListener

try:
    from kafka import KafkaProducer
except Exception:
    KafkaProducer = None


class OnlineForecaster(SlidingWindowListener):
    """
    OnlineForecaster: lightweight, streaming-friendly forecaster that mirrors
    the feature extraction and prediction logic used in the offline notebook.

    Designed to be registered as an observer with the existing
    `KafkaStreamReader` in `read_kafka_stream.py` (it implements the same
    `on_new_window_<topic>` callbacks). It loads models from `./artifacts`
    (relative to `consumer_node`) and emits predictions to an optional
    Kafka topic `forecasts` as well as printing them. The InfluxDB writer
    will still call `predict()` on the same instance when available and
    include predictions in writes.
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

    # Singleton support (keeps same instance between DatabaseWriter and main)
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, artifacts_dir=None, horizons=(1, 2, 3), verb=False, max_buffer_size=50,
                 lag_hours=(1, 2, 3, 6, 12, 24), roll_windows=(3, 6, 12), db_writer=None,
                 required_samples=None):
        # allow reconfiguration without recreating background threads
        already_init = getattr(self, '_initialized', False)
        if already_init:
            # update settings if provided
            if verb is not None:
                self.verb = verb
            if horizons is not None:
                self.horizons = list(horizons)
            if lag_hours is not None:
                self.lag_hours = tuple(lag_hours)
            if roll_windows is not None:
                self.roll_windows = tuple(roll_windows)
            if max_buffer_size is not None:
                self.max_buffer_size = max_buffer_size
            if db_writer is not None:
                self.db_writer = db_writer
            return

        super().__init__()
        self.verb = verb
        self.horizons = list(horizons)
        self.lag_hours = tuple(lag_hours)
        self.roll_windows = tuple(roll_windows)
        self.max_buffer_size = max_buffer_size
        self.db_writer = db_writer
        # required_samples can be provided to reduce the minimum number of
        # historical records required before predictions are attempted.
        # If not provided, it defaults to max(lag_hours)+1 (historical behavior).
        if required_samples is None:
            # allow ENV override as well
            env_rs = os.getenv('FORECAST_REQUIRED_SAMPLES')
            try:
                self.required_samples = int(env_rs) if env_rs is not None else None
            except Exception:
                self.required_samples = None
        else:
            try:
                self.required_samples = int(required_samples)
            except Exception:
                self.required_samples = None

        if artifacts_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.artifacts_dir = os.path.join(current_dir, 'artifacts')
        else:
            self.artifacts_dir = artifacts_dir

        self.data_buffer = {}  # topic -> deque
        self.buffer_lock = Lock()

        self.models = {}
        self.last_predictions = {}

        self._load_models()

        # optional Kafka producer for publishing forecasts
        self.kafka_topic = os.getenv('FORECASTS_TOPIC', 'forecasts')
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.producer = None
        if KafkaProducer is not None:
            try:
                self.producer = KafkaProducer(bootstrap_servers=self.kafka_servers,
                                              value_serializer=lambda v: json.dumps(v).encode('utf-8'))
            except Exception:
                self.producer = None

    def _load_models(self):
        try:
            files = os.listdir(self.artifacts_dir)
        except Exception:
            files = []

        import re
        pattern = re.compile(r"^model_(.+)_H(\d+)_hgb\.joblib$")
        for fn in files:
            m = pattern.match(fn)
            if not m:
                continue
            target_name = m.group(1)
            h = int(m.group(2))
            model_path = os.path.join(self.artifacts_dir, fn)
            try:
                mdl = joblib.load(model_path)
                self.models.setdefault(target_name, {})[f"H{h}"] = mdl
            except Exception as e:
                if self.verb:
                    print(f"Error loading model {model_path}: {e}")
                    traceback.print_exc()

        if self.verb:
            print("OnlineForecaster loaded models:", {k: list(v.keys()) for k, v in self.models.items()})

    def _update_buffer(self, topic, value):
        with self.buffer_lock:
            if topic not in self.data_buffer:
                self.data_buffer[topic] = deque(maxlen=self.max_buffer_size)
            try:
                if value == -200 or value == "-200":
                    val = np.nan
                else:
                    val = float(value)
            except Exception:
                val = np.nan
            self.data_buffer[topic].append(val)

    def _get_required_samples(self, target_name):
        if getattr(self, 'required_samples', None) is not None:
            return int(self.required_samples)
        return max(self.lag_hours) + 1

    def _extract_features(self, target=None):
        target_key = target if target is not None else next(iter(self.models.keys()), None)
        if target_key is None:
            return None
        target_topic = self.COLUMN_TO_TOPIC.get(target_key)
        required = self._get_required_samples(target_key)
        with self.buffer_lock:
            if target_topic not in self.data_buffer or len(self.data_buffer[target_topic]) < required:
                return None
            data = {}
            for topic, col in self.TOPIC_TO_COLUMN.items():
                vals = list(self.data_buffer.get(topic, []))
                if len(vals) >= required:
                    values = vals[-required:]
                else:
                    pad = [np.nan] * (required - len(vals))
                    values = pad + vals
                data[col] = values

        timestamps = pd.date_range(end=datetime.now(), periods=required, freq='h')
        df = pd.DataFrame(data, index=timestamps)
        # time features
        df_time = pd.DataFrame({
            'hour': timestamps.hour,
            'dow': timestamps.dayofweek,
            'month': timestamps.month,
            'is_weekend': (timestamps.dayofweek >= 5).astype(int)
        }, index=timestamps)
        df = pd.concat([df, df_time], axis=1)

        # build feature dict
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ['hour', 'dow', 'month', 'is_weekend']]
        feature_dict = {}
        for c in num_cols:
            for L in self.lag_hours:
                feature_dict[f"{c}_lag{L}"] = df[c].shift(L)
            for W in self.roll_windows:
                feature_dict[f"{c}_rollmean{W}"] = df[c].shift(1).rolling(W, min_periods=1).mean()
                feature_dict[f"{c}_rollstd{W}"] = df[c].shift(1).rolling(W, min_periods=1).std()

        feature_dict['hour'] = df_time['hour']
        feature_dict['dow'] = df_time['dow']
        feature_dict['month'] = df_time['month']
        feature_dict['is_weekend'] = df_time['is_weekend']

        X = pd.DataFrame(feature_dict, index=df.index)
        if len(X) == 0:
            return None
        return X.iloc[-1:].copy()

    def predict(self, horizon=None, target=None):
        target_key = target if target is not None else next(iter(self.models.keys()), None)
        if target_key is None:
            return None
        X = self._extract_features(target_key)
        if X is None or len(X) == 0:
            return None
        preds = {}
        hs = [horizon] if horizon is not None else self.horizons
        models_for_target = self.models.get(target_key, {})
        for h in hs:
            key = f'H{h}'
            if key in models_for_target:
                model = models_for_target[key]
                try:
                    fn = None
                    if hasattr(model, 'feature_names_in_'):
                        fn = model.feature_names_in_
                    elif hasattr(model, 'named_steps'):
                        for step in model.named_steps.values():
                            if hasattr(step, 'feature_names_in_'):
                                fn = step.feature_names_in_
                                break
                    if fn is not None:
                        Xord = X.reindex(columns=fn, fill_value=np.nan)
                    else:
                        Xord = X
                    p = model.predict(Xord)[0]
                    preds[f'H+{h}'] = float(p)
                except Exception as e:
                    if self.verb:
                        print(f"Error predicting {target_key} H+{h}: {e}")
                        traceback.print_exc()
                    preds[f'H+{h}'] = None
            else:
                preds[f'H+{h}'] = None

        # store
        self.last_predictions[target_key] = preds

        # publish to Kafka if possible
        try:
            if self.producer is not None:
                msg = {
                    'target': target_key,
                    'predictions': preds,
                    'time': datetime.utcnow().isoformat() + 'Z'
                }
                self.producer.send(self.kafka_topic, msg)
                # best-effort flush for low-latency
                self.producer.flush(timeout=1)
        except Exception:
            if self.verb:
                print('Failed to publish forecast to Kafka')

        if self.verb:
            print(f"OnlineForecaster: {target_key} -> {preds}")

        return preds

    # sliding window callbacks
    def _make_on_new_window(topic_key):
        def handler(self, data):
            if not data:
                return
            latest = data[-1]
            try:
                self._update_buffer(topic_key, latest['value'])
            except Exception:
                pass
            target_name = self.TOPIC_TO_COLUMN.get(topic_key)
            preds = self.predict(target=target_name)
            # Write predictions to InfluxDB if we have a DB writer and a valid
            # measurement timestamp in the message key (same format used elsewhere).
            try:
                if preds and getattr(self, 'db_writer', None) is not None and 'key' in latest:
                    try:
                        # pass topic_key (e.g. 'pt08_s1_co') and measurement time string
                        self.db_writer.write_prediction(topic_key, preds, latest['key'])
                    except Exception:
                        if self.verb:
                            print('OnlineForecaster: failed to write_prediction to InfluxDB')
            except Exception:
                pass

            if preds and self.verb:
                print(f"OnlineForecaster (cb) {target_name} -> {preds}")
        return handler

    # generate the same named callbacks as the old OfflineForecaster so
    # it can be registered interchangeably with KafkaStreamReader.
    on_new_window_pt08_s1_co = _make_on_new_window('pt08_s1_co')
    on_new_window_pt08_s2_nmhc = _make_on_new_window('pt08_s2_nmhc')
    on_new_window_pt08_s3_nox = _make_on_new_window('pt08_s3_nox')
    on_new_window_pt08_s4_no2 = _make_on_new_window('pt08_s4_no2')
    on_new_window_pt08_s5_o3 = _make_on_new_window('pt08_s5_o3')
    on_new_window_t = _make_on_new_window('t')
    on_new_window_rh = _make_on_new_window('rh')
    on_new_window_ah = _make_on_new_window('ah')
