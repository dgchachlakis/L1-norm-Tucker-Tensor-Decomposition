import numpy as np
from scipy import linalg
from utils import *
def bitflipping(X, K, Qinit = [], tol = 1e-6):
    D = X.shape[0]
    N = X.shape[1]
    if len(Qinit) == 0:
        Q = linalg.orth(np.random.randn(D, K))
    else:
        Q = Qinit
    metric_across_iter = [metricL1pca(X, Q)]
    identityN = np.identity(N)
    identityK = np.identity(K)
    while True:  
        B = np.sign(X.T @ Q)
        temp = np.zeros((N, K))
        for n in range(N):
            for k in range(K):
                B_candidate = B-2*B[n, k]*np.outer(identityN[:, n] , identityK[:, k].T)
                temp[n, k] = np.linalg.norm(X @ B_candidate, ord = 'nuc')
        nbest, kbest = np.where(temp == temp.max() )
        nbest = nbest[0]
        kbest = kbest[0]
        Bnew = B-2*B[nbest, kbest]*np.outer(identityN[:, nbest] , identityK[:, kbest].T)
        Qnew = procrustes(X @ Bnew)

        if metricL1pca(X, Qnew)-metric_across_iter[-1] < tol:
            break
        else:
            Q = Qnew
            B = Bnew
            metric_across_iter.append(metricL1pca(X, Q))
    return Q, B, metric_across_iter