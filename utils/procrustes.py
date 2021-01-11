import numpy as np
def procrustes(A):
    K = A.shape[1]
    U, S, Vt = np.linalg.svd(A)
    return U[:, :K] @ Vt
    
    