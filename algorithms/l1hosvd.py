import tensorly as tl
import l1pca 
import numpy as np
from scipy import linalg
def l1hosvd(tensor, tensor_ranks, Qinit = [], solver = "fixedpoint", tol = 1e-6):
    ndimensions = tensor.ndim
    tensor_shape = tensor.shape
    if len(Qinit) == 0:
        factors = []
        for n in range(ndimensions):
            factors.append(linalg.orth(np.random.randn(tensor_shape[n], tensor_ranks[n])))
    else:
        factors = Qinit
    for n in range(ndimensions):
        nunfolding = tl.unfold(tensor, n)
        if solver == 'fixedpoint':
            factors[n] = l1pca.fixedpoint(nunfolding, tensor_ranks[n], factors[n], tol = tol)[0]
        elif solver == 'bitflipping':
            factors[n] = l1pca.bitflipping(nunfolding, tensor_ranks[n], factors[n], tol = tol)[0]
        elif solver == 'exactpoly':
            factors[n] = l1pca.exactpoly(nunfolding, tensor_ranks[n])[0]
        else: 
            factors[n] = l1pca.exact(nunfolding, tensor_ranks[n])[0]
    core = tl.tenalg.multi_mode_dot(tensor, factors, transpose = True)
    return core, factors