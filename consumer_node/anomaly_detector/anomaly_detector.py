import numpy as np

from global_statistics.StreamStatistics import SimpleTDigest

from listeners.sliding_window_listener import SlidingWindowListener
from InfluxDB.InfluxDbUtilities import DatabaseWriter

from anomaly_detector.novelty_function import derivateNoveltyFn
from anomaly_detector.outlier_estimator import MissingValueDetector, WindowOutlierDetector, TDigestOutlierDetector
from anomaly_detector.adwin import ADWIN


class InWindowAnomalyDetector(SlidingWindowListener):
    def __init__(self,
                 dbWriter:DatabaseWriter,
                 novelty_fn=derivateNoveltyFn,
                 estimators:dict={
                    "global": TDigestOutlierDetector(tdigest=SimpleTDigest(delta=.1)),
                    "local": WindowOutlierDetector(),
                    "missing":  MissingValueDetector(),
                 },
                 min_samples=8,
                 verb=False,
                 enable_drift_detection=True,
                 adwin_delta=0.002):
        super().__init__()

        self.dbWriter = dbWriter
        self.novelty_fn = novelty_fn
        self.estimators = estimators
        self.min_samples = min_samples
        self.verb = verb
        self.enable_drift_detection = enable_drift_detection

        self.adwin_detectors = {}
        self.adwin_delta = adwin_delta

        self.observers = []

        if self.verb:
            print("-------------------------------")
            print("InWindowAnomalyDetector started")
            if self.enable_drift_detection:
                print("ADWIN concept drift detection: ENABLED")
            print("-------------------------------")

    def fit(self, data):
        self.data_ = data
        return self

    def look(self):
        x = self.data_[-1]

        self.x_key_ = x["key"]
        self.x_value_ = float(x["value"])
        self.x_topic_ = x["topic"]
        self.x_ = x

        return self
    
    def transform(self):
        values = np.array([float(item["value"]) for item in self.data_], dtype=float)
        values = np.where(values == -200, np.nan, values)
        novelty_scores = self.novelty_fn(values)

        self.values_ = values
        self.novelty_scores_ = novelty_scores

        self.X_nov_train_ = novelty_scores[:-1]
        self.x_nov_ = novelty_scores[-1]

        return self
    
    def detect_drift(self):
        if not self.enable_drift_detection:
            return False
        
        if len(self.novelty_scores_) < self.min_samples:
            return False
        
        if self.x_topic_ not in self.adwin_detectors:
            self.adwin_detectors[self.x_topic_] = ADWIN(
                delta=self.adwin_delta,
                min_window_length=self.min_samples
            )
        
        adwin = self.adwin_detectors[self.x_topic_]
        
        if not np.isnan(self.x_nov_):
            drift_detected = adwin.update(self.x_nov_)
            
            if drift_detected:
                window_size = adwin.get_window_size()
                if self.verb:
                    print(f"\n[ADWIN] Concept drift detected in topic '{self.x_topic_}' "
                          f"at observation {self.x_key_} (novelty score: {self.x_nov_:.4f})")
                    print(f"[ADWIN] Window size before drift: {window_size}")
                
                self.dbWriter.write_drift(
                    sample=self.x_,
                    topic=self.x_topic_,
                    novelty_score=self.x_nov_,
                    window_size=window_size
                )
                
                self._reset_estimators_after_drift()
                
                return True
        
        return False
    
    def _reset_estimators_after_drift(self):
        for est_key, estimator in self.estimators.items():
            if isinstance(estimator, TDigestOutlierDetector):
                if hasattr(estimator.tdigest, 'reset'):
                    estimator.tdigest.reset()
                else:
                    estimator.tdigest = SimpleTDigest(delta=.1)
                if self.verb:
                    print(f"[ADWIN] Reset estimator: {est_key}")
        
        if self.verb:
            print(f"[ADWIN] Estimators reset/adapted for topic '{self.x_topic_}'")

    def detect(self):
        if (len(self.data_) < self.min_samples):
            print(f"Minimum {self.min_samples} data points are required for anomaly detector. Skipping detection.")
            return self

        if self.verb:
            print(f"Detecting anomalies in Topic (\"{self.x_topic_}\") within a window with length={len(self.data_)}).")
            print(f"Window elements: " + ", ".join(f"{item['key']}: {item['value']}" for item in self.data_))
            print(f"Predicting whether [{self.x_key_} with value of {self.x_value_}] is anomalous.")

        drift_detected = self.detect_drift()
        
        predictions = {}

        for (est_key, estimator) in self.estimators.items():

            if isinstance(estimator, TDigestOutlierDetector):
                y_hat = estimator.predict(self.x_nov_)
            else:
                y_hat = estimator.fit(self.X_nov_train_) \
                    .predict(self.x_nov_)
            
            if y_hat:
                predictions[est_key] = 1

        is_anomalous = any(predictions.values())

        if (is_anomalous):
            estimator_keys = [key for key, val in predictions.items() if val == 1]

            if self.verb:
                self.print_detected(self.values_, self.novelty_scores_, self.x_key_, self.x_nov_, estimator_keys)

            self.dbWriter.write_anomaly(self.x_, types=estimator_keys, topic=self.x_topic_)
        
        return self

    def update_estimators(self):
        for (est_key, estimator) in self.estimators.items():
            if hasattr(estimator, "update"):
                estimator.update(self.x_nov_)

        return self

    def on_new_window_pt08_s1_co(self, data):
        self.fit(data).look().transform().detect().update_estimators()

    def on_new_window_pt08_s2_nmhc(self, data):
        self.fit(data).look().transform().detect().update_estimators()

    def on_new_window_pt08_s3_nox(self, data):
        self.fit(data).look().transform().detect().update_estimators()

    def on_new_window_pt08_s4_no2(self, data):
        self.fit(data).look().transform().detect().update_estimators()

    def on_new_window_pt08_s5_o3(self, data):
        self.fit(data).look().transform().detect().update_estimators()

    def on_new_window_t(self, data):
        self.fit(data).look().transform().detect().update_estimators()

    def on_new_window_ah(self, data):
        self.fit(data).look().transform().detect().update_estimators()

    def on_new_window_rh(self, data):
        self.fit(data).look().transform().detect().update_estimators()
    
    def on_new_window_co_gt(self, data):
        pass

    def on_new_window_nmhc_gt(self, data):
        pass

    def on_new_window_c6h6_gt(self, data):
        pass

    def on_new_window_nox_gt(self, data):
        pass

    def on_new_window_no2_gt(self, data):
        pass

    def print_detected(self, values, novelty_scores, x_key, x_nov_score, estimator_keys):
        print("")
        print("\t" + "-"*50)
        print(f"\t Observation [{x_key} with novelty score of {x_nov_score}] is anomalous by estimator(s) {estimator_keys}.")
        print("\n\n\t Window:")
        print("\n\t Values:\t\t", values)
        print("\n\t Novelty scores:\t", novelty_scores)
        print("\n\t" + "-"*50)
        print("")
        print("")
