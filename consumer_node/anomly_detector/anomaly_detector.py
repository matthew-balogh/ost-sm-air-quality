import numpy as np

from listeners.sliding_window_listener import SlidingWindowListener

np.set_printoptions(legacy='1.25', suppress=True)

# FIXME: move?
def calc_IQR(data):
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    return IQR, Q1, Q3

class InWindowAnomalyDetector(SlidingWindowListener):
    def __init__(self, verb=False):
        super().__init__()
        self.verb = verb
        self.observers = []
        self.most_recently_left_behind = {}

        if self.verb:
            print("-------------------------------")
            print("InWindowAnomalyDetector started")
            print("-------------------------------")

    def prepare(self, topic, data):
        if self.verb:
            print(f"Detecting anomalies in Topic (\"{topic}\") within a window with length={len(data)}).")

        mrlf = self.most_recently_left_behind.get(topic, np.nan)
        values = np.array([float(item['value']) for item in data])
        values_diff = np.diff(values, prepend=mrlf)

        if self.verb:
            print(f"\t Sensor measurements in window: {values}")
            print(f"\t Differences from previous: {values_diff}")

        return values, values_diff
    

    def predict(self, obs_key, obs_score, values_diff):
        if self.verb:
            print(f"Predicting whether [{obs_key} with diff value of {obs_score}] is anomalous or not within the window.")

        # window statistics
        values_diff_IQR, values_diff_Q1, values_diff_Q3 = calc_IQR(values_diff)

        # detect peaks (ignore falls for now, FIXME later)
        pred = (obs_score > 0) & (obs_score > (values_diff_Q3 + 1.5 * values_diff_IQR))

        return pred

    def on_new_window_co_gt(self, data):
        if self.verb:
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))


        obs_topic = data[-1]['topic']
        obs_key = data[-1]['key']

        values, values_diff = self.prepare(obs_topic, data)
        obs_diff = values_diff[-1]

        is_anomalous = self.predict(obs_key, obs_diff, values_diff)

        # TODO: silence?
        if is_anomalous:
            print("")
            print("\t ---")
            print(f"\t Observation [{obs_key} with diff value of {obs_diff}] is anomalous.")
            print("\t ---")
            print("")
            print("")

        # TODO:
        # send event to downstream processes (post-processing such as model updating, and live visualization)

        self.most_recently_left_behind[obs_topic] = values[0]

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
