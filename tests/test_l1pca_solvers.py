import l1pca 
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def test_l1pca_exact(D=3,N=3,K=2):
    X=np.random.randn(D,N)
    Qopt, Bopt = l1pca.exact(X, K, verbose=True)
    return Qopt,Bopt

def test_l1pca_exactpoly(D=3,N=3,K=2):
    X=np.random.randn(D,N)
    Qopt, Bopt = l1pca.exactpoly(X, K, verbose=True)
    return Qopt,Bopt

def test_l1pca_bitflipping(D=3,N=3,K=2):
    X=np.random.randn(D,N)
    Q, metric_evolution=l1pca.bitflipping(X,K,verbose=True)
    return Q, metric_evolution

def test_l1pca_fixedpoint(D=3,N=3,K=2):
    X=np.random.randn(D,N)
    Q, metric_evolution=l1pca.fixedpoint(X,K,verbose=True)
    return Q, metric_evolution

def compare_l1solvers():
    D=4
    N=5
    matrix_rank=3;
    matrix=np.random.randn(D,N)
    Qin=linalg.orth(np.random.randn(D,matrix_rank))
    
    Qexact= l1pca.exact(matrix, matrix_rank, verbose=False)[0]
    mexact=np.sum(np.abs((matrix.T @ Qexact).flatten()))
    Qexactpoly= l1pca.exactpoly(matrix, matrix_rank, verbose=False)[0]
    mexactpoly=np.sum(np.abs((matrix.T @ Qexactpoly).flatten()))
    Qfp, mfp= l1pca.fixedpoint(matrix, matrix_rank, Qin, verbose=False)
    Qbf, mbf= l1pca.bitflipping(matrix, matrix_rank, Qin, verbose=False)

    xax_fp=list(range(len(mfp)))
    xax_bf=list(range(len(mbf)))
    xmax=np.max([len(xax_fp), len(xax_bf)])

    plt.plot(xax_fp,mfp, label="Fixed-point")
    plt.plot(xax_bf,mbf, label="Bit-flipping")
    plt.plot([0, xmax],[mexactpoly,mexactpoly], '--', label="Exact-poly")
    plt.plot([0, xmax],[mexact,mexact], ':', label="Exact")
    plt.ylabel('L1-PCA metric')
    plt.xlabel('Iteration index')
    plt.legend()
    plt.show()
