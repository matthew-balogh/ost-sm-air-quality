
import numpy as np

class MovingSequentialStatistics:
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
            self.squaredMean = self.squaredMean + (new_sample**2 - self.squaredMean)/self.count;
        if self.method == "exponential":
            self.squaredMean = (1 - self.alpha) * self.squaredMean + self.alpha * new_sample**2;
    
    def VarianceUpdate(self):
        self.variance = (self.count * (self.squaredMean - self.mean**2))/(self.count - 1)

    def UpdateAll(self,new_sample):
        self.count += 1;
        self.meanUpdate(new_sample);
        self.SquaredmeanUpdate(new_sample);
        self.VarianceUpdate();





class WindowSequentialStatistics:
    def __init__(self, window_size = 30):
        self.mean = 0;
        self.squaredMean = 0;
        self.variance = 0;
        self.count = 0
        self.window_size = window_size;
        self.window = np.zeros(self.window_size);

    def meanUpdate(self):
        self.mean = np.mean(self.window);
        

    def SquaredMeanUpdate(self):
        self.squaredMean = np.mean(self.window**2);

    def VarianceUpdate(self):
        self.variance = self.squaredMean - self.mean**2;




    def UpdateAll(self,new_sample):
        self.count += 1;
        self.window[(self.count % self.window_size) - 1] = new_sample;

        if self.count >= self.window_size:
            self.meanUpdate();
            self.SquaredmeanUpdate();
            self.VarianceUpdate();
        else:
            print("N is still smaller than window size");
