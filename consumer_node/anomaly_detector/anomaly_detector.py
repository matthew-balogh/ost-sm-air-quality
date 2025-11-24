import numpy as np
import global_statistics.IQR as IQR

from abc import ABC, abstractmethod
from listeners.sliding_window_listener import SlidingWindowListener
from global_statistics.StreamStatistics import SimpleTDigest


class PeakPickingStrategy(ABC):
    @abstractmethod
    def predict(self, novelty_score, novelty_fn): pass

    @abstractmethod
    def update(self, x): pass
    

class TDigestOutlierStrategy(PeakPickingStrategy):
    def __init__(self, delta=.1, iqr_fence=1.5):
        super().__init__()
        self.delta = delta
        self.iqr_fence = iqr_fence
        self.algorithm_ = SimpleTDigest(delta)

    def predict(self, novelty_score, novelty_fn):
        q1, q3 = self.algorithm_.percentile(25), self.algorithm_.percentile(75)
        if (q1 is not None) and (q3 is not None):
            iqr = (q3 - q1)
            is_anomalous = (novelty_score > (q3 + self.iqr_fence * iqr))
        else: is_anomalous = False

        return {
            "pred": is_anomalous,
            "type": "global",
            "message": "Outlier based on the global opinion (T-digest)."
        }
    
    def update(self, x):
        return self.algorithm_.update(x)

class WindowOutlierStrategy(PeakPickingStrategy):
    def __init__(self, iqr_fence=1.5):
        super().__init__()
        self.iqr_fence = iqr_fence

    def predict(self, novelty_score, novelty_fn):
        iqr, q1, q3 = IQR.calc(novelty_fn)
        is_anomalous = (novelty_score > (q3 + self.iqr_fence * iqr))

        return {
            "pred": is_anomalous,
            "type": "local",
            "message": "Outlier based on the local opinion (within the window)."
        }
    
    def update(self, x):
        pass

class MissingValueStrategy(PeakPickingStrategy):
    def __init__(self):
        super().__init__()

    def predict(self, novelty_score, novelty_fn):
        is_anomalous = np.isnan(novelty_score)

        return {
            "pred": is_anomalous,
            "type": "missing",
            "message": "Value / novelty score is missing."
        }

    def update(self, x):
        pass
    
def derivateNoveltyFn(input):
    if np.all(np.isnan(input)):
        return np.full_like(input, np.nan)
    
    median = np.nanmedian(input)
    nan_mask = np.isnan(input)

    _input = np.where(nan_mask, median, input)
    diff = np.diff(_input, prepend=input[0])
    diff[nan_mask] = np.nan
    diff[diff < 0] = 0
    return diff


class InWindowAnomalyDetector(SlidingWindowListener):
    def __init__(self,
                 strategies:list[PeakPickingStrategy]=[WindowOutlierStrategy(), TDigestOutlierStrategy(), MissingValueStrategy()],
                 novelty_fn=derivateNoveltyFn,
                 verb=False):
        super().__init__()

        self.strategies = strategies
        self.novelty_fn = novelty_fn
        self.verb = verb

        self.observers = []

        if self.verb:
            print("-------------------------------")
            print("InWindowAnomalyDetector started")
            print("-------------------------------")

    # TODO: replicate to other methods / merge methods
    def on_new_window_co_gt(self, data):
        obs_index = -1
        topic = data[obs_index]["topic"]


        if self.verb:
            print(f"Detecting anomalies in Topic (\"{topic}\") within a window with length={len(data)}).")
            print(f"Window elements: " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
            print(f"Predicting whether [{data[obs_index]['key']} with value of {data[obs_index]['value']}] is anomalous.")


        values = np.array([float(item["value"]) for item in data], dtype=float)
        values = np.where(values == -200, np.nan, values) # TODO: move upstream
        novelty_scores = self.novelty_fn(values)
        nov_score = novelty_scores[obs_index]

        predictions = []

        for s in self.strategies:
            pred = s.predict(nov_score, novelty_scores)
            predictions.append(pred)
            s.update(nov_score)

        is_anomalous = any(p["pred"] for p in predictions)

        if (is_anomalous):
            types = [p["type"] for p in predictions if p["pred"]]
            reasons = [p["message"] for p in predictions if p["pred"]]

            print("")
            print("\t" + "-"*50)
            print(f"\t Observation [{data[obs_index]['key']} with novelty score of {nov_score}] is anomalous.")
            print(f"\n\t\t Reason(s):")
            for t, r in zip(types, reasons):
                print(f"\t\t * TYPE: {t}\t| {r}")
            print("\n\n\t Window:")
            print("\n\t Values:\t\t", values)
            print("\n\t Novelty scores:\t", novelty_scores)
            print("\n\t" + "-"*50)
            print("")
            print("")

        return is_anomalous, predictions

    def on_new_window_pt08_s1_co(self, data):
        if self.verb:
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))

    def on_new_window_nmhc_gt(self, data):
        if self.verb:
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))

    def on_new_window_c6h6_gt(self, data):
        if self.verb:
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))

    def on_new_window_pt08_s2_nmhc(self, data):
        if self.verb:
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))

    def on_new_window_nox_gt(self, data):
        if self.verb:
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))

    def on_new_window_pt08_s3_nox(self, data):
        if self.verb:
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))

    def on_new_window_no2_gt(self, data):
        if self.verb:
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))

    def on_new_window_pt08_s4_no2(self, data):
        if self.verb:
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))

    def on_new_window_pt08_s5_o3(self, data):
        if self.verb:
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))

    def on_new_window_t(self, data):
        if self.verb:
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))

    def on_new_window_ah(self, data):
        if self.verb:
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))

    def on_new_window_rh(self, data):
        if self.verb:
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))
