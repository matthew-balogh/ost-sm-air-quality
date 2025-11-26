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
                 verb=False):
        super().__init__()

        self.dbWriter = dbWriter
        self.novelty_fn = novelty_fn
        self.estimators = estimators
        self.verb = verb

        self.observers = []

        if self.verb:
            print("-------------------------------")
            print("InWindowAnomalyDetector started")
            print("-------------------------------")

    def detect(self, data):
        if (len(data) < 8):
            print("Minimum 8 data points are required for anomaly detector. Skipping detection.")
            return

        x = data[-1]
        topic = x["topic"]

        if self.verb:
            print(f"Detecting anomalies in Topic (\"{topic}\") within a window with length={len(data)}).")
            print(f"Window elements: " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
            print(f"Predicting whether [{x['key']} with value of {x['value']}] is anomalous.")

        # ensure data format
        values = np.array([float(item["value"]) for item in data], dtype=float)
        values = np.where(values == -200, np.nan, values) # TODO: move upstream?

        # novelty function
        novelty_scores = self.novelty_fn(values)

        # predictions
        predictions = {}

        for (est_key, estimator) in self.estimators.items():
            X_train = novelty_scores[:-1]
            x_test = novelty_scores[-1]

            if hasattr(estimator, "update"):
                y_hat = estimator.predict(x_test)
                estimator.update(x_test)
            else:
                y_hat = estimator.fit(X_train) \
                    .predict(x_test)
            
            if y_hat:
                predictions[est_key] = 1

        is_anomalous = any(predictions.values())

        if (is_anomalous):
            estimator_keys = [key for key, val in predictions.items() if val == 1]

            if self.verb:
                self.print_detected(values, novelty_scores, x['key'], x_test, estimator_keys)

            # store in db
            self.dbWriter.write_anomaly(x, types=estimator_keys, topic=topic)


    # detect anomalies in sensor data

    def on_new_window_pt08_s1_co(self, data):
        self.detect(data)

    def on_new_window_pt08_s2_nmhc(self, data):
        self.detect(data)

    def on_new_window_pt08_s3_nox(self, data):
        self.detect(data)

    def on_new_window_pt08_s4_no2(self, data):
        self.detect(data)

    def on_new_window_pt08_s5_o3(self, data):
        self.detect(data)

    def on_new_window_t(self, data):
        self.detect(data)

    def on_new_window_ah(self, data):
        self.detect(data)

    def on_new_window_rh(self, data):
        self.detect(data)
    
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
