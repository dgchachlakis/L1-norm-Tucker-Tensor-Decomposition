import l1tucker 
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def test_l1hosvd():
    tensor=np.random.randn(3,3,3)
    tensor_ranks=np.array([2,2,2])

    core_fp,factors_fp=l1tucker.l1hosvd(tensor,tensor_ranks,solver='fixed-point',verbose=True,tol=1e-4)
    core_bf,factors_bf=l1tucker.l1hosvd(tensor,tensor_ranks,solver='bit-flipping',verbose=True,tol=1e-4)
    core,factors=l1tucker.l1hosvd(tensor,tensor_ranks,solver='exact-poly',verbose=True,tol=1e-4)

def test_l1hooi():
    tensor=np.random.randn(3,3,3)
    tensor_ranks=np.array([2,2,2])

    core,factors,metric=l1tucker.l1hooi(tensor, tensor_ranks, solver='fixed-point', verbose=False, tol=1e-4)
    core,factors,metric=l1tucker.l1hooi(tensor, tensor_ranks, solver='bit-flipping', verbose=False, tol=1e-4)
    core,factors,metric=l1tucker.l1hooi(tensor, tensor_ranks, solver='exact-poly', verbose=False, tol=1e-4)

    plt.plot(range(len(metric)),metric)
    plt.ylabel('L1-Tucker metric')
    plt.xlabel('Iteration index')
    plt.show()


