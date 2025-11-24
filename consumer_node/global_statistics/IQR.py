import numpy as np

def calc(data):
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    return IQR, Q1, Q3
