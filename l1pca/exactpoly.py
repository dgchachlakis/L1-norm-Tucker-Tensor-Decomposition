import numpy as np
import itertools
import math
from utils import *
def exactpoly(X, K):
    D = X.shape[0]
    N = X.shape[1]
    U, S, Vt = np.linalg.svd(X, full_matrices = False)
    matrix_rank = np.linalg.matrix_rank(X)
    Q = np.diag(S[:matrix_rank])@Vt[:matrix_rank, :]
    Bcandidates = compute_candidates(Q, halfSphere = True)
    Qopt = np.zeros((D, K))
    Bopt = np.zeros((N, K))
    metopt = 0
    if K == 1:
        numOfCandidates = Bcandidates.shape[1]
        for n in range(numOfCandidates):
            b = Bcandidates[:, n, None]
            met = np.linalg.norm(X @ b)
            if met > metopt:
                Qopt = X @ b / np.linalg.norm(X @ b)
                metopt = metricL1pca(X, Qopt)
                Bopt = b
    else:
        for idx in itertools.product(range(Bcandidates.shape[1]), repeat = K):
            Bcand = Bcandidates[:, np.array(idx).astype(np.int)]
            met = np.linalg.norm(X @ Bcand , ord = 'nuc')
            if met > metopt:
                Qopt = procrustes(X @ Bcand)
                Bopt = Bcand
                metopt = metricL1pca(X, Qopt)
    return Qopt, Bopt