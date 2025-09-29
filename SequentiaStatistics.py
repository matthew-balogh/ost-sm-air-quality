
class SequentialStatistics:
    def __init__(self, method = "sequential", alpha = "0.5"):
        self.mean = 0;
        self.squaredMean = 0;
        self.variance = 0;
        self.count = 0
        self.method = method;
        self.alpha = alpha;

    def meanUpdate(self, new_sample):
        if self.method == "sequential":
            self.mean = self.mean + (new_sample - self.mean)/self.count;
        if self.method == "exponential":
            self.mean = (1 - self.alpha) * self.mean + self.alpha * new_sample;
    
    def SquaredMeanUpdate(self, new_sample):
        if self.method == "sequential":
            self.squaredMean = self.squaredMean + (new_sample - self.squaredMean)/self.count;
        if self.method == "exponential":
            self.squaredMean = (1 - self.alpha) * self.squaredMean + self.alpha * new_sample;
    
    def VarianceUpdate(self):
        self.variance = (self.count * (self.squaredMean - self.mean^2))/(self.count - 1)

    def UpdateAll(self,new_sample):
        self.count += 1;
        self.meanUpdate(new_sample);
        self.SquaredmeanUpdate(new_sample);
        self.VarianceUpdate();

        