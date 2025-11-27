import numpy as np

from global_statistics.StreamStatistics import SimpleTDigest

from listeners.sliding_window_listener import SlidingWindowListener
from InfluxDB.InfluxDbUtilities import DatabaseWriter

from anomaly_detector.novelty_function import derivateNoveltyFn
from anomaly_detector.outlier_estimator import MissingValueDetector, WindowOutlierDetector, TDigestOutlierDetector


class InWindowAnomalyDetector(SlidingWindowListener):
    def __init__(self,
                 dbWriter:DatabaseWriter,
                 novelty_fn=derivateNoveltyFn,
                 estimators:dict={
                    "global": TDigestOutlierDetector(tdigest=SimpleTDigest(delta=.1), upper_only=True),
                    "local": WindowOutlierDetector(upper_only=True),
                    "missing":  MissingValueDetector(),
                 },
                 min_samples=8,
                 verb=False):
        super().__init__()

        self.dbWriter = dbWriter
        self.novelty_fn = novelty_fn
        self.estimators = estimators
        self.min_samples = min_samples
        self.verb = verb

        self.observers = []

        if self.verb:
            print("-------------------------------")
            print("InWindowAnomalyDetector started")
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
        values = np.where(values == -200, np.nan, values) # TODO: move upstream?
        novelty_scores = self.novelty_fn(values)

        self.values_ = values
        self.novelty_scores_ = novelty_scores

        self.X_nov_train_ = novelty_scores[:-1]
        self.x_nov_ = novelty_scores[-1]

        return self

    def detect(self):
        if (len(self.data_) < self.min_samples):
            print(f"Minimum {self.min_samples} data points are required for anomaly detector. Skipping detection.")
            return self

        if self.verb:
            print(f"Detecting anomalies in Topic (\"{self.x_topic_}\") within a window with length={len(self.data_)}).")
            print(f"Window elements: " + ", ".join(f"{item['key']}: {item['value']}" for item in self.data_))
            print(f"Predicting whether [{self.x_key_} with value of {self.x_value_}] is anomalous.")

        # predictions
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

            # store in db
            self.dbWriter.write_anomaly(self.x_, types=estimator_keys, topic=self.x_topic_)
        
        return self

    def update_estimators(self):
        for (est_key, estimator) in self.estimators.items():
            if hasattr(estimator, "update"):
                estimator.update(self.x_nov_)

        return self


    # detect anomalies in sensor data

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
    
    # skip anomaly detection on reference data
    
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
