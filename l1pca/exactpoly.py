import numpy as np
import itertools
import math

def exactpoly(X, K, half_sphere=True, verbose=False):
    
    def phi(A):
        K=A.shape[1]
        U, S, Vt = np.linalg.svd(A)
        return U[:,:K] @ Vt

    def l1pca_metric(X, Q):
        return np.sum(np.abs((X.T @ Q).flatten()))

    def mysign(x,tol=1e-7):
        x[np.abs(x)<tol]=0
        return np.sign(x)

    def bin_array_1D(num, m):
        #Convert a positive integer num into an m-bit bit vector
        return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

    def bin_array_2D(N):    
        L=2**N
        B=np.zeros((L,N))
        for n in range(L):
            B[n,:]=bin_array_1D(n, N)
        return 2*B-1

    def compute_candidates(Q, half_sphere=True):
        bDict = {}
        d=Q.shape[0]
        N=Q.shape[1]
        numOfambiguities=d-1
        Bpool=bin_array_2D(numOfambiguities)
        combinations=list(itertools.combinations((range(N)), d-1))
        for combination in combinations:
            Qbar=Q[:,combination]
            U,S,Vt = np.linalg.svd(Qbar, full_matrices=True)
            c=U[:,-1]
            b_ambiguous=mysign(Q.T @ c).flatten()
            for n in range(Bpool.shape[0]):
                b=b_ambiguous.copy()
                try:
                    b[b==0]=Bpool[n,:].copy()
                except:
                    print('Error: Check tolerance of mysign function. The program exits.')
                    exit()
                b=b[0]*b
                signature=tuple(b)
                if signature not in bDict:
                    bDict[signature]=[val for val in combination]
                else:
                    value=bDict[signature]
                    for v in combination:
                        if v not in value:
                            bDict[signature].append(v) 
        if not half_sphere:
            bDict_otherHalf={}
            for b in bDict:
                b_current=np.array(b)
                values_current=bDict[b]
                bDict_otherHalf[tuple(-b_current)]=[val for val in values_current]
            bDict.update(bDict_otherHalf)

        return bDict


    D=X.shape[0]
    N=X.shape[1]

    U,S,Vt=np.linalg.svd(X,full_matrices=False)
    matrix_rank=np.linalg.matrix_rank(X)
    Q=np.diag(S[:matrix_rank])@Vt[:matrix_rank,:]
    Bcandidates=compute_candidates(Q,half_sphere)

    Qopt=np.zeros((D,K))
    Bopt=np.zeros((N,K))
    metopt=0

    if K==1:
        for btuple in Bcandidates:
            b=np.array(btuple)
            met=np.linalg.norm(X @ b )
            if met>metopt:
                Qopt=X @ b / np.linalg.norm(X @ b)
                metopt=l1pca_metric(X, Qopt)
                Bopt=b
        number_of_signatures_half_sphere=0
        for n in range(matrix_rank):
            number_of_signatures_half_sphere=number_of_signatures_half_sphere+math.comb(N-1,n)

        if verbose:
            print('Number of unique candidates (half-sphere): \t' + str(number_of_signatures_half_sphere))
            print('Number of candidates examined: \t\t\t' + str(len(Bcandidates)))
    else:
        BcandidatesMatrix=np.zeros((N,len(Bcandidates)))
        for i,btuple in enumerate(Bcandidates):
            b=np.array(btuple)
            BcandidatesMatrix[:,i]=b
        for idx in itertools.product(range(BcandidatesMatrix.shape[1]), repeat = K):
            Bcand=BcandidatesMatrix[:,np.array(idx).astype(np.int)]
            met=np.linalg.norm(X @ Bcand ,ord='nuc')
            if met>metopt:
                Qopt=phi(X @ Bcand)
                Bopt=Bcand
                metopt=l1pca_metric(X, Qopt)
   

    return Qopt, Bopt