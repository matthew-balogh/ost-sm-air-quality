import numpy as np
import math
from collections import deque


class ADWIN:
    
    def __init__(self, delta=0.002, min_window_length=5, max_buckets=5):
        self.delta = delta
        self.min_window_length = min_window_length
        self.max_buckets = max_buckets
        
        self.buckets = deque()
        self.total_count = 0
        self.total_sum = 0.0
        
        self.drift_detected = False
        self.last_drift_index = None
        
    def update(self, value):
        self._add_bucket(value)
        
        drift_detected = self._detect_drift()
        
        if drift_detected:
            self.drift_detected = True
            self.last_drift_index = self.total_count
        else:
            self.drift_detected = False
            
        return drift_detected
    
    def _add_bucket(self, value):
        self.buckets.append(deque([(value, 1)]))
        self.total_count += 1
        self.total_sum += value
        
        self._compress_buckets()
    
    def _compress_buckets(self):
        i = 0
        while i < len(self.buckets):
            if (i + 1 < len(self.buckets) and 
                len(self.buckets[i]) + len(self.buckets[i+1]) <= self.max_buckets):
                
                while self.buckets[i+1]:
                    self.buckets[i].append(self.buckets[i+1].popleft())
                del self.buckets[i+1]
            else:
                i += 1
    
    def _detect_drift(self):
        if self.total_count < self.min_window_length:
            return False
        
        n0 = 0
        sum0 = 0.0
        
        for i, bucket in enumerate(self.buckets):
            for sum_val, count in bucket:
                n0 += count
                sum0 += sum_val
                
                n1 = self.total_count - n0
                sum1 = self.total_sum - sum0
                
                if n0 >= self.min_window_length and n1 >= self.min_window_length:
                    mean0 = sum0 / n0
                    mean1 = sum1 / n1
                    
                    if self._cut_point(n0, n1, mean0, mean1):
                        self._drop_oldest(n0)
                        return True
        
        return False
    
    def _cut_point(self, n0, n1, mean0, mean1):
        if n0 == 0 or n1 == 0:
            return False
        
        variance = 1.0 / (n0 + n1)
        
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        epsilon = math.sqrt((2.0 / m) * math.log(2.0 / self.delta))
        
        diff = abs(mean0 - mean1)
        
        return diff >= epsilon
    
    def _drop_oldest(self, n_to_drop):
        dropped = 0
        while dropped < n_to_drop and self.buckets:
            bucket = self.buckets[0]
            if len(bucket) == 0:
                self.buckets.popleft()
                continue
            
            sum_val, count = bucket.popleft()
            
            if dropped + count <= n_to_drop:
                self.total_count -= count
                self.total_sum -= sum_val
                dropped += count
            else:
                partial = n_to_drop - dropped
                remaining = count - partial
                self.total_count -= partial
                self.total_sum -= sum_val * (partial / count)
                bucket.appendleft((sum_val * (remaining / count), remaining))
                dropped = n_to_drop
            
            if len(bucket) == 0:
                self.buckets.popleft()
    
    def get_mean(self):
        if self.total_count == 0:
            return 0.0
        return self.total_sum / self.total_count
    
    def get_window_size(self):
        return self.total_count
    
    def reset(self):
        self.buckets.clear()
        self.total_count = 0
        self.total_sum = 0.0
        self.drift_detected = False
        self.last_drift_index = None
