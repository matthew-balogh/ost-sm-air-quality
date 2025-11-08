from listeners.sliding_window_listener import SlidingWindowListener


class InWindowAnomalyDetector(SlidingWindowListener):
    def __init__(self, verb=False):
        super().__init__()
        self.verb = verb
        self.observers = []

    def on_new_window_co_gt(self, data):
        if self.verb:
            print(f"len={len(data)} | " + ", ".join(f"{item['key']}: {item['value']}" for item in data))

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

    def predict(self, observation, lag):
        """
        Predicts whether the given `observation` is anomalous or not based on the given `lag` array
        """

        if self.verb:
            print(f"Predicting whether [{observation}] is anomalous or not, using a lag of {lag.shape}.")

        pred = True

        if self.verb:
            print(pred)

        return pred