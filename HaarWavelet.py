import numpy as np
from scipy.stats import cumfreq 

def w1(timeseries):
    '''Compute w1, the Haar wavelet transform of the lowest scale.
    w1 is just a fancy term for memory-less differences between adjacent points in the timeseries
    
    Args: 
        timeseries (np.array): time series of the data    
    '''
    #print(np.diff(timeseries, n=1))
    return np.diff(timeseries, n=1)

def sdevFromW1(w1):
    '''Compute the standard deviation from a w1 Haar wavelet. Sdev*sqrt(2) is 68.2% from zero for the cumulative distribution of abs(w1)'''
    sortedAbsW1 = np.sort(np.abs(w1))
    if(len(sortedAbsW1)>1):
        sdev = sortedAbsW1[int(np.round(0.682*len(w1)))] / np.sqrt(2)
        return sdev
    else:
        return 0