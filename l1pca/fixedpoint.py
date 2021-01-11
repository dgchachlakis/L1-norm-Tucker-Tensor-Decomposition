import numpy as np
from utils import *
def fixedpoint(X, K, Qinit = [], tol = 1e-6):
    D = X.shape[0]
    N = X.shape[1]
    if len(Qinit) == 0:
        Q = np.linalg.orth(np.random.randn(D, K))
    else:
        Q = Qinit
    metric_across_iter = [metricL1pca(X, Q)]
    while True:
        Q = procrustes(X @ np.sign(X.T @ Q))
        metric_across_iter.append(metricL1pca(X, Q))        
        if metric_across_iter[-1]-metric_across_iter[-2] < tol:
            break
    B = np.sign(X.T @ Q)    
    return Q, B, metric_across_iter