import numpy as np
import bisect

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


## when a new sample arrives, find the closest centrois using distance.
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




from math import ceil
from typing import Iterable, Dict, Tuple, List






class LossyCounting:
    """
    Lossy Counting for singleton items.

    Parameters
    ----------
    epsilon : float
        Error parameter (0 < epsilon < 1). Bucket width w = ceil(1/epsilon).
    """
    def __init__(self, epsilon: float):
        assert 0 < epsilon < 1, "epsilon must be in (0,1)"
        self.epsilon = epsilon
        self.w = ceil(1.0 / epsilon)           # bucket width
        self.N = 0                             # total items seen so far
        self.bucket_id = 1                     # current bucket id (starts at 1)
        # table: item -> (count, delta)
        self.table: Dict[object, Tuple[int, int]] = {}

    def _maybe_prune(self):
        # prune at bucket boundaries
        if self.N % self.w != 0:
            return
        b = self.bucket_id
        # delete entries with count + delta <= bucket_id
        to_delete = [item for item, (c, delta) in self.table.items() if c + delta <= b]
        for item in to_delete:
            del self.table[item]
        # after pruning, next bucket id
        self.bucket_id += 1

    def process_item(self, item):
        """Process a single item from the stream."""
        self.N += 1
        if item in self.table:
            c, delta = self.table[item]
            self.table[item] = (c + 1, delta)
        else:
            # new entry: count=1, delta = current_bucket_id - 1
            self.table[item] = (1, self.bucket_id - 1)

        # prune if the condition frequency + Delta <= current_bucket_id)
        self._maybe_prune()


    def estimated_counts(self) -> Dict[object, int]:
        """Return estimated counts (lower bounds) for all items currently in table."""
        return {item: c for item, (c, _) in self.table.items()}
    


    def frequent_items(self, support: float) -> List[Tuple[object, int]]:
        """
        Return list of (item, est_count) that are frequent wrt support s.

        Papers state items with true frequency > s*N are guaranteed present.
        To be conservative we return items with est_count >= (support - epsilon) * N.
        """
        s = support
        assert 0 <= s <= 1
        threshold = max(0, (s - self.epsilon) * self.N)
        result = []
        for item, (c, delta) in self.table.items():
            if c >= threshold:
                result.append((item, c))
        # sort descending by estimated count
        result.sort(key=lambda x: -x[1])
        return result

    def size(self) -> int:
        """Number of entries currently stored (space usage indicator)."""
        return len(self.table)


