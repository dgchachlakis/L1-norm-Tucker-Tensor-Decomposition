import numpy as np
from scipy import linalg

def fixedpoint(X, K, Qinit=[], verbose=False, tol=1e-6):
    
    def phi(A):
        K=A.shape[1]
        U, S, Vt = np.linalg.svd(A)
        return U[:,:K] @ Vt

    def l1pca_metric(X, Q):
        return np.sum(np.abs((X.T @ Q).flatten()))

    D=X.shape[0]
    N=X.shape[1]

    if len(Qinit)==0:
        Q=linalg.orth(np.random.randn(D,K))
    else:
        Q=Qinit
    
    metric_across_iter=[l1pca_metric(X, Q)]
    
    if verbose:
        print('Iteration \tMetric')
        it=0
    
    while True:
        
        if verbose:
            print(str(it) + '\t' + str(metric_across_iter[-1]))
            it+=1

        Q=phi(X @ np.sign(X.T @ Q))
        metric_across_iter.append(l1pca_metric(X, Q))        
        if metric_across_iter[-1]-metric_across_iter[-2]<tol:
            break
           
    return Q, metric_across_iter