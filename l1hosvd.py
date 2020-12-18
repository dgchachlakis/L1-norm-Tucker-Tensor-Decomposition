import tensorly as tl
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

def l1hosvd(tensor, tensor_ranks, Qinit=[],solver="fixedpoint", verbose=False, tol=1e-6):
    ndimensions=tensor.ndim
    tensor_shape=tensor.shape

    if len(Qinit)==0:
        factors=[]
        for n in range(ndimensions):
            factors.append(linalg.orth(np.random.randn(tensor_shape[n],tensor_ranks[n])))
    else:
        factors=Qinit

    for n in range(ndimensions):
        nunfolding=tl.unfold(tensor,n)
        factors[n],junk=l1pca(nunfolding,tensor_ranks[n],factors[n])

    core=tl.tenalg.multi_mode_dot(tensor, factors, transpose=True)

    return core, factors

def l1pca(X, K, Qinit=[], solver="fixedpoint", verbose=False, tol=1e-6):
    
    D=X.shape[0]
    N=X.shape[1]

    if len(Qinit)==0:
        Q=linalg.orth(np.random.randn(D,K))
    else:
        Q=Qinit
    
    if solver=='fixedpoint':
        metric_across_iter=[]
        metric_across_iter.append(l1pca_metric(X, Q))
        while True:
            Qnew=phi(X @ np.sign(X.T @ Q))
            mnew=l1pca_metric(X, Qnew)
            if mnew-metric_across_iter[-1]>tol:
                Q=Qnew
                metric_across_iter.append(l1pca_metric(X, Q))
            else:
                break
        if verbose==True:
            plt.plot(range(len(metric_across_iter)),metric_across_iter)
            plt.ylabel('L1-PCA metric')
            plt.xlabel('Iteration index')
            plt.show()
        return Q, metric_across_iter
    elif solver=='bit-flipping':
        pass # to be implemented
    elif solver=='exact-exhaustive':
        pass # to be implemented
    elif solver=='exact-poly':
        pass # to be implemented
    else:
        pass

def phi(A):
    K=A.shape[1]
    U, S, Vt = linalg.svd(A)
    return U[:,:K] @ Vt

def l1pca_metric(X, Q):
    return np.sum(np.abs((X.T @ Q).flatten()))