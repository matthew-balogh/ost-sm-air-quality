import numpy as np
import bisect
import math
from scipy.stats import norm

class MovingStatistics:

    def __init__(self, method = "sequential", alpha = 0.3):
        self.mean = 0;
        self.squaredMean = 0;
        self.variance = 0;
        self.count = 0
        self.method = method;
        self.alpha = alpha;

    def meanUpdate(self, new_sample):

        if self.method == "sequential":
            self.mean = self.mean + (new_sample - self.mean)/self.count;
        
        elif self.method == "exponential":
            self.mean = (1 - self.alpha) * self.mean + self.alpha * new_sample;
        
        else:
            raise NotImplementedError("This method is not implemented.")
    
    def SquaredMeanUpdate(self, new_sample):
        if self.method == "sequential":
            self.squaredMean = self.squaredMean + (new_sample**2 - self.squaredMean)/self.count;
        elif self.method == "exponential":
            self.squaredMean = (1 - self.alpha) * self.squaredMean + self.alpha * new_sample**2;
        else:
            raise NotImplementedError("This method is not implemented.")
    
    def VarianceUpdate(self):
        if self.count >= 2:
            self.variance = (self.count * (self.squaredMean - self.mean**2))/(self.count - 1)

    def UpdateAll(self,new_sample):
        self.count += 1;
        self.meanUpdate(new_sample);
        self.SquaredMeanUpdate(new_sample);
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
            self.SquaredMeanUpdate();
            self.VarianceUpdate();
        else:
            print("N is still smaller than window size");


## when a new sample arrives, find the closest centroid using distance.
## Check the compression limit ???()
## if it doesn’t violate the compression limit, merge it: new_mean, new weight.
## Merging centroids: scan through centroids in order:
## if adjacent centroids with a combined weight (number of sample) <= delta * total weight merge them.
## then update the mean of the merged centroid.
class SimpleTDigest:

    def __init__(self, delta=0.01):
        self.centroids = []   # List of Centroids, sorted by mean
        self.total_weight = 0
        self.delta = delta    # Compression factor (smaller = more accuracy)

    def update(self, x):
        """Add a new point x to the digest"""
        c = Centroid(x)
        means = [cent.mean for cent in self.centroids]
        idx = bisect.bisect_left(means, x)
        self.centroids.insert(idx, c)
        self.total_weight += 1
        self.compress()

    def compress(self):
        """Merge adjacent centroids to limit the number of centroids"""
        merged = []
        i = 0
        while i < len(self.centroids):
            c = self.centroids[i]
            while i + 1 < len(self.centroids) and (c.weight + self.centroids[i+1].weight) <= self.delta * self.total_weight:
                next_c = self.centroids[i+1]
                # Weighted mean merge
                new_mean = (c.mean * c.weight + next_c.mean * next_c.weight) / (c.weight + next_c.weight)
                c = Centroid(new_mean, c.weight + next_c.weight)
                i += 1
            merged.append(c)
            i += 1
        self.centroids = merged

    def percentile(self, q):
        """Approximate the q-th percentile"""
        
        if not self.centroids:
            return None
        

        target = q / 100 * self.total_weight
        cumulative = 0
        for c in self.centroids:
            cumulative += c.weight
            if cumulative >= target:
                return c.mean
        return self.centroids[-1].mean

class Centroid:
    def __init__(self, mean, weight=1):
        self.mean = mean
        self.weight = weight



## T-digest algorithm for inter-quartile

##@misc{dunning2019computingextremelyaccuratequantiles,
#       title={Computing Extremely Accurate Quantiles Using t-Digests}, 
#       author={Ted Dunning and Otmar Ertl},
#       year={2019},
#       eprint={1902.04023},
#       archivePrefix={arXiv},
#       primaryClass={stat.CO},
#       url={https://arxiv.org/abs/1902.04023}, 
# }


class MKTrendDetector:
    def __init__(self,t_digest, quantile_step = 5):
        self.t_digest = t_digest;
        self.quantile_probs = [i/100 for i in range(quantile_step, 100, quantile_step)]  # 0.05, 0.10, ..., 0.95
        self.quantile_values = [None]*len(self.quantile_probs) 
        self.length = 0;
        self.S = 0;
        self.Z = 0;
        self.P = 0;
        self.min_samples = len(self.quantile_probs);
    
    def calculate_quantiles_tdigest(self,sample):
        # First we should get a decent number of samples
        self.length+= 1;

        self.t_digest.update(sample);
        
        # if self.length <= self.min_samples:
        #     print("Not enough samples for Trend detection");
        # else:
        for idx, p in enumerate(self.quantile_probs):
            self.quantile_values[idx] = self.t_digest.percentile(p);
    
    def approx_cdf(self, x):     
        '''
        Approximate the CDF function, and interpolate. <=>  calculate P(X<= xi) = F(xi)
        '''     

        ## Extreme cases
        if x <= self.quantile_values[0]:
            return 0.0
        if x >= self.quantile_probs[-1]:
            return 1.0
        
        for i in range(len(self.quantile_values)-1):
            q_i = self.quantile_values[i]
            q_next = self.quantile_values[i+1]
            p_i = self.quantile_probs[i]
            p_next = self.quantile_probs[i+1]

            if q_i <= x <= q_next:
                return p_i + (x - q_i)/(q_next - q_i) * (p_next - p_i)
            
        return 1.0
    


    def update(self, x):
        """
        Update the online MK statistic with a new value x
        """
        # Step 1: update quantiles (may use small history initially)

        self.calculate_quantiles_tdigest(x);
        
                
        if self.length <= self.min_samples:
            print("Not enough samples for Trend detection");
        # Step 2: compute new St by adding the sample contribution to St-1. 
        else:
            if self.length > 0: 
                cdf = self.approx_cdf(x);
                self.S += self.length * (2*cdf - 1)

        # Step 3: update count
        self.length += 1;                

            
        
    def compute_variance_Z(self):
        # Let's assume no ties.
        varS = self.length * (self.length-1) * (2*self.length + 5) / 18
        if self.S > 0:
            self.Z = (self.S - 1) / math.sqrt(varS)
        elif self.S < 0:
            self.Z = (self.S + 1) / math.sqrt(varS)
        else:
            self.Z = 0.0;



    def p_value(self):
        self.compute_variance_Z()
        self.P = 2 * (1 - norm.cdf(abs(self.Z)))  # two-sided



    def trend(self, alpha=0.05):
        
        self.p_value();

        if self.P < alpha:

            if self.S > 0:
                return "Significant increasing trend"
            else:
                return "Significant decreasing trend"
            
        else:
            return "No significant trend"

        
        
    
