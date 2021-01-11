import numpy as np
import itertools
import math
from utils import *
def exact(X, K):
    D = X.shape[0]
    N = X.shape[1]
    Bexst = decimal2binary(list(range(2**N)), N);
    Qopt = np.zeros((D, K))
    Bopt = np.zeros((N, K))
    metopt = 0
    if K == 1:
        for n in range(Bexst.shape[1]):
            b = Bexst[:, n, None]
            met = np.linalg.norm(X @ b)
            if met > metopt:
                Qopt = X @ b / np.linalg.norm(X @ b)
                metopt = metricL1pca(X, Qopt)
                Bopt = b
    else:
        for idx in itertools.product(range(Bexst.shape[1]), repeat = K):
            Bcand = Bexst[:, np.array(idx).astype(np.int)]
            met = np.linalg.norm(X @ Bcand , ord = 'nuc')
            if met > metopt:
                Qopt = procrustes(X @ Bcand)
                Bopt = Bcand
                metopt = metricL1pca(X, Qopt)
    return Qopt, Bopt