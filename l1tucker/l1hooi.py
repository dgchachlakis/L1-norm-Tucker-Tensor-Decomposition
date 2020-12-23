import tensorly as tl
import l1pca 
import numpy as np
from scipy import linalg


def l1hooi(tensor, tensor_ranks, Qinit=[], solver="fixed-point", verbose=False, tol=1e-6):

    def l1tucker_metric(tensor,factors):
        core=tl.tenalg.multi_mode_dot(tensor, factors, modes=list(range(tensor.ndim)), transpose=True)
        return np.sum(np.abs(core.flatten()))

    tensor_shape=tensor.shape
    dimensions=tensor.ndim
    # Initialization
    if len(Qinit)==0:
        factors=[]
        for n in range(dimensions):
            factors.append(linalg.orth(np.random.randn(tensor_shape[n],tensor_ranks[n])))
    else:
        factors=Qinit

    # l1hooi
    metric_across_iterations=[l1tucker_metric(tensor,factors)]
    if verbose:
        print('Iteration \tMetric')
        it=0

    while True:
        
        if verbose:
            print(str(it) + '\t' + str(metric_across_iterations[-1]))
            it+=1

        for n in range(tensor.ndim):
            modes=list(range(n))+list(range(n+1,tensor.ndim))
            tensorAn=tl.tenalg.multi_mode_dot(tensor, factors[:n]+factors[n+1:], modes=modes, transpose=True)
            unfoldingAn=tl.unfold(tensorAn,n)
            if solver=='fixed-point':
                factors[n]=l1pca.fixedpoint(unfoldingAn,tensor_ranks[n],factors[n], tol=tol)[0]
            elif solver=='bit-flipping':
                factors[n]=l1pca.bitflipping(unfoldingAn,tensor_ranks[n],factors[n], tol=tol)[0]
            elif solver=='exact-poly':
                factors[n]=l1pca.exactpoly(unfoldingAn,tensor_ranks[n])[0]
            else:
                factors[n]=l1pca.exact(unfoldingAn,tensor_ranks[n])[0]

        metric_across_iterations.append(l1tucker_metric(tensor,factors))
        if metric_across_iterations[-1]-metric_across_iterations[-2]<tol:
            break
    
    core=tl.tenalg.multi_mode_dot(tensor, factors, modes=list(range(tensor.ndim)), transpose=True)
    return core, factors, metric_across_iterations