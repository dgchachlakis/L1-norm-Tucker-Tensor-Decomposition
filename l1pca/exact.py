import numpy as np
import itertools
import math

def exact(X, K, verbose=False):
    
    def phi(A):
        K=A.shape[1]
        U, S, Vt = np.linalg.svd(A)
        return U[:,:K] @ Vt

    def l1pca_metric(X, Q):
        return np.sum(np.abs((X.T @ Q).flatten()))

    def bin_array_1D(num, m):
        #Convert a positive integer num into an m-bit bit vector
        return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

    def bin_array_2D(N):    
        L=2**N
        B=np.zeros((L,N))
        for n in range(L):
            B[n,:]=bin_array_1D(n, N)
        return 2*B-1

    D=X.shape[0]
    N=X.shape[1]
    Bcandidates=bin_array_2D(N);

    Qopt=np.zeros((D,K))
    Bopt=np.zeros((N,K))
    metopt=0

    if K==1:
        for n in range(Bcandidates.shape[0]):
            b=Bcandidates[n,:]
            met=np.linalg.norm(X @ b )
            if met>metopt:
                Qopt=X @ b / np.linalg.norm(X @ b)
                metopt=l1pca_metric(X,Qopt)
                Bopt=b
    else:
        for idx in itertools.product(range(Bcandidates.shape[0]), repeat = K):
            Bcand=Bcandidates[np.array(idx).astype(np.int),:].T
            met=np.linalg.norm(X @ Bcand ,ord='nuc')
            if met>metopt:
                Qopt=phi(X @ Bcand)
                Bopt=Bcand
                metopt=l1pca_metric(X,Qopt)
   

    return Qopt, Bopt