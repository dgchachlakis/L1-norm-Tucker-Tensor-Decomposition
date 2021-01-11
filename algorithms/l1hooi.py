import tensorly as tl
import l1pca 
import numpy as np
from utils import *
from scipy import linalg
def l1hooi(tensor, tensor_ranks, Qinit = [], solver = "fixedpoint", tol = 1e-6):
    tensor_shape = tensor.shape
    dimensions = tensor.ndim
    if len(Qinit) == 0:
        factors = []
        for n in range(dimensions):
            factors.append(linalg.orth(np.random.randn(tensor_shape[n], tensor_ranks[n])))
    else:
        factors = Qinit
    metric_across_iterations = [metricL1tucker(tensor, factors)]
    while True:
        for n in range(tensor.ndim):
            modes = list(range(n))+list(range(n+1, tensor.ndim))
            tensorAn = tl.tenalg.multi_mode_dot(tensor, factors[:n]+factors[n+1:], modes = modes, transpose = True)
            unfoldingAn = tl.unfold(tensorAn, n)
            if solver == 'fixedpoint':
                factors[n] = l1pca.fixedpoint(unfoldingAn, tensor_ranks[n], factors[n], tol = tol)[0]
            elif solver == 'bitflipping':
                factors[n] = l1pca.bitflipping(unfoldingAn, tensor_ranks[n], factors[n], tol = tol)[0]
            elif solver == 'exactpoly':
                factors[n] = l1pca.exactpoly(unfoldingAn, tensor_ranks[n])[0]
            else:
                factors[n] = l1pca.exact(unfoldingAn, tensor_ranks[n])[0]
        metric_across_iterations.append(metricL1tucker(tensor, factors))
        if metric_across_iterations[-1]-metric_across_iterations[-2] < tol:
            break
    core = tl.tenalg.multi_mode_dot(tensor, factors, modes = list(range(tensor.ndim)), transpose = True)
    return core, factors, metric_across_iterations