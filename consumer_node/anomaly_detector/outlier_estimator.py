import numpy as np
import global_statistics.IQR as IQR

from global_statistics.StreamStatistics import SimpleTDigest

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class BaseOutlierDetector(BaseEstimator):
    pass

class MissingValueDetector(BaseOutlierDetector):
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.isnan(np.asarray(X))

class WindowOutlierDetector(BaseOutlierDetector):
    def __init__(self, iqr_fence=1.5, upper_only=False):
        self.iqr_fence = iqr_fence
        self.upper_only = upper_only

    def fit(self, X, y=None):
        self.IQR_, self.Q1_, self.Q3_ = IQR.calc(np.asarray(X))
        return self

    def predict(self, X):
        check_is_fitted(self)
        lower = self.Q1_ - self.iqr_fence * self.IQR_
        upper = self.Q3_ + self.iqr_fence * self.IQR_
        X_arr = np.asarray(X)

        if self.upper_only:
            is_outlier = X_arr > upper
        else:
            is_outlier = (X_arr < lower) | (X_arr > upper)

        return is_outlier

class TDigestOutlierDetector(BaseOutlierDetector):
    def __init__(self, tdigest:SimpleTDigest, iqr_fence=1.5, upper_only=False):
        self.tdigest = tdigest
        self.iqr_fence = iqr_fence
        self.upper_only = upper_only

    def update(self, X):
        self.tdigest.update(X)

    def predict(self, X):
        Q1, Q3 = self.tdigest.percentile(25), self.tdigest.percentile(75)

        # Return all False if TDigest has no data
        if Q1 is None or Q3 is None:
            return np.zeros_like(np.asarray(X), dtype=bool)
        
        IQR = (Q3 - Q1)

        lower = Q1 - self.iqr_fence * IQR
        upper = Q3 + self.iqr_fence * IQR
        X_arr = np.asarray(X)

        if self.upper_only:
            is_outlier = X_arr > upper
        else:
            is_outlier = (X_arr < lower) | (X_arr > upper)

        return is_outlier
