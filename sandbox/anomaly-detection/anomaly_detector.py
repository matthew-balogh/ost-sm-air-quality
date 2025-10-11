from listeners.sliding_window_listener import SlidingWindowListener

class InWindowAnomalyDetector():

    def __init__(self, verb=False):
        self.verb = verb

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