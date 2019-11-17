import scipy
import numpy as np


def ncc(a, b, sweep=False):
    """
    Compute the normalized cross-correlation between two signals:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6147431/#sec001title
    
    Be careful when using this function with zero-arrays (returns 0)
    """
    eps = np.finfo(np.float).eps
    corr = scipy.signal.correlate(a, b)
    if not sweep:
        mid = ()
        for d in corr.shape:
            i = np.floor((d - 1) / 2).astype(int)
            mid += ((slice(i, i + 2) if d % 2 == 0 else [i]),)
        corr = corr[mid].mean(axis=0)
    norm = np.sqrt((a ** 2).sum() * (b ** 2).sum()) + eps
    return corr / norm
