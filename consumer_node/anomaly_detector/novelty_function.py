import numpy as np

def derivateNoveltyFn(data:list, clip_negative=True):
    if np.all(np.isnan(data)):
        return data
    
    median = np.nanmedian(data)
    nan_mask = np.isnan(data)

    _data = np.where(nan_mask, median, data)
    diff = np.diff(_data, prepend=data[0])
    diff[nan_mask] = np.nan

    if clip_negative:
        diff[diff < 0] = 0

    return diff
