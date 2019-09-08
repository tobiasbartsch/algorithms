import pymc3 as pm
import numpy as np
from scipy.stats import multivariate_normal 

def parametric_Neiswanger(traces):
    '''Approximate each subposterior aus Gaussian; then construct the combined posterior density (
        also approximated as Gaussian).
    
    Args:
        traces: list of traces, each produced by a different machine for a different batch of the data.
    
    Returns:
        means_total, cov_total: Means and covariance matrix of the approximated combined posterior.
    '''

    means = []
    covs = []

    for trace in traces:
        df = pm.trace_to_dataframe(trace)
        means.append(np.mean(df, axis=0))
        covs.append(np.cov(df, rowvar=0))

    cov_total = np.linalg.inv(np.sum(np.array([np.linalg.inv(cov) for cov in covs]), axis=0))
    #return means, covs, cov_total
    means_total = np.dot(np.sum(np.array([np.dot(np.linalg.inv(cov), np.array(mean)) for cov, mean in zip(covs, means)]), axis=0), cov_total)
    
    return means_total, cov_total





def sample_non_parametric_Neiswanger(traces):
    '''Asymptotically exact posterior sampling with nonparametric density product estimation.
    Follows algorithm 1 in Neiswanger 2014
    
    Args:
        traces: list of traces, each produced by a different machine for a different batch of the data.
    
    Returns:
        np.array of samples from the approximated combined posterior.
    '''

    #covert traces to dataframes:
    dflist = []
    for trace in traces:
        dflist.append(pm.trace_to_dataframe(trace).copy())
    
    samples = []
    
    #dimensionality of parameter space
    d = len(dflist[0].columns)
    
    #number of traces
    M = len(traces)

    #number of samples per trace
    T = len(dflist[0])
    
    #1: Define a starting mixture: One random index for of each of the M traces.
    t = np.random.randint(low=0, high=T, size=M)
    
    #2: loop through 0 to the length of a trace:
    for i in np.arange(T):
        if(i%1000 == 0):
            print(i)
        j = i+1
        #3: set bandwidth
        h = j**(-1/(4+d)) #Why this is a function of i is unclear to me.
        #4: Loop through the chains
        for m in np.arange(M):
            c = t.copy()
            c[m] = np.random.randint(low=0, high=T)
            u = np.random.uniform(low=0, high=1)
            thetas_t = np.array([np.array(df.iloc[index]) for index, df in zip(t,dflist)])
            thetas_c = np.array([np.array(df.iloc[index]) for index, df in zip(c,dflist)])
            #thetas_t = np.array([np.array(list(trace[index].values())) for index, trace in zip(t,traces)])
            #thetas_c = np.array([np.array(list(trace[index].values())) for index, trace in zip(c,traces)])
            
            #get weights
            wc = weight(thetas_c, h, d)
            wt = weight(thetas_t, h, d)
            if u < wc/wt:
                t = c.copy()
        thetas_t = np.array([np.array(df.iloc[index]) for index, df in zip(t,dflist)])
        thetas_mean = np.mean(thetas_t, axis=0)
        cov = (h**2)/M * np.identity(d)
        sample = np.random.multivariate_normal(mean=thetas_mean, cov=cov)
        samples.append(sample)
    
    return np.array(samples)
            
def weight(thetas, h, d):
    '''equation 6 in Neiswanger 2014'''
    thetas_mean = np.mean(thetas, axis=0)
    probs = [multivariate_normal.pdf(theta, mean=thetas_mean, cov=h**2 * np.identity(d)) for theta in thetas]
    return np.cumprod(probs)[-1]