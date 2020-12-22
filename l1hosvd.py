import tensorly as tl
from l1pca_fixedpoint import *
from l1pca_bitflipping import *
from l1pca_exactpoly import *

def l1hosvd(tensor, tensor_ranks, Qinit=[], solver="fixedpoint", verbose=False, tol=1e-6):
    ndimensions=tensor.ndim
    tensor_shape=tensor.shape

    if len(Qinit)==0:
        factors=[]
        for n in range(ndimensions):
            factors.append(linalg.orth(np.random.randn(tensor_shape[n],tensor_ranks[n])))
    else:
        factors=Qinit

    for n in range(ndimensions):
        if verbose:
            print('Computing mode-'+str(n)+' unfolding...')
        nunfolding=tl.unfold(tensor,n)
        if verbose:
            print('Solving l1-pca of mode-'+str(n)+' unfolding...')
        if solver=='fixed-point':
            factors[n]=l1pca_fixedpoint(nunfolding, tensor_ranks[n], factors[n], tol=tol)[0]
        elif solver=='bit-flipping':
            factors[n]=l1pca_bitflipping(nunfolding, tensor_ranks[n], factors[n], tol=tol)[0]
        elif solver=='exact-poly':
            factors[n]=l1pca_exactpoly(nunfolding, tensor_ranks[n])[0]
        else: 
            #factors[n]=l1pca_exact(nunfolding, tensor_ranks[n])[0]
            raise Exception('To be defined shortly')
        if verbose:
            print('Done...')
    core=tl.tenalg.multi_mode_dot(tensor, factors, transpose=True)

    return core, factors