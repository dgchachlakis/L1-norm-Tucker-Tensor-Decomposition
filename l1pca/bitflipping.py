import numpy as np
from scipy import linalg
def bitflipping(X, K, Qinit=[], verbose=False, tol=1e-6):
    
    def phi(A):
        K=A.shape[1]
        U, S, Vt = np.linalg.svd(A)
        return U[:,:K] @ Vt

    def l1pca_metric(X, Q):
        return np.sum(np.abs((X.T @ Q).flatten()))

    def norm_nuc(A):
        return np.sum(np.linalg.svd(A)[1].flatten())

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
    
    identityN=np.identity(N)
    identityK=np.identity(K)
    

    while True:
        
        if verbose:
            print(str(it) + '\t' + str(metric_across_iter[-1]))
            it+=1
        
        B=np.sign(X.T @ Q)
        temp=np.zeros((N,K))
        for n in range(N):
            for k in range(K):
                B_candidate=B-2*B[n,k]*np.outer(identityN[:,n] , identityK[:,k].T)
                temp[n,k]=norm_nuc(X @ B_candidate)
        nbest,kbest = np.where(temp == temp.max() )
        nbest=nbest[0]
        kbest=kbest[0]
        
        Bnew=B-2*B[nbest,kbest]*np.outer(identityN[:,nbest] , identityK[:,kbest].T)
        Qnew=phi(X @ Bnew)

        if l1pca_metric(X, Qnew)-metric_across_iter[-1]<tol:
            break
        else:
            Q=Qnew
            B=Bnew
            metric_across_iter.append(l1pca_metric(X, Q))
    
    return Q, metric_across_iter