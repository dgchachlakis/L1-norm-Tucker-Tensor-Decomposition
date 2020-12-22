from l1hooi import *
import matplotlib.pyplot as plt

tensor=np.random.randn(3,3,3)
tensor_ranks=np.array([2,2,2])

core,factors,metric=l1hooi(tensor, tensor_ranks, solver='fixed-point', verbose=False, tol=1e-4)
core,factors,metric=l1hooi(tensor, tensor_ranks, solver='bit-flipping', verbose=False, tol=1e-4)
core,factors,metric=l1hooi(tensor, tensor_ranks, solver='exact-poly', verbose=False, tol=1e-4)
#core,factors,metric=l1hooi(tensor, tensor_ranks, solver='exact', verbose=False, tol=1e-4)


plt.plot(range(len(metric)),metric)
plt.ylabel('L1-Tucker metric')
plt.xlabel('Iteration index')
plt.show()

