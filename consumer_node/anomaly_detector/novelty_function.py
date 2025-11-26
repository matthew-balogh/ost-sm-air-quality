import numpy as np

def derivateNoveltyFn(data:list):
    if np.all(np.isnan(data)):
        return data
    
    median = np.nanmedian(data)
    nan_mask = np.isnan(data)

    _data = np.where(nan_mask, median, data)
    diff = np.diff(_data, prepend=data[0])
    diff[nan_mask] = np.nan

    return diff
