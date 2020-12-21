from l1hooi import *
import matplotlib.pyplot as plt

tensor=np.random.randn(8,8,8)
tensor_ranks=np.array([3,3,3])

core,factors,metric=l1hooi(tensor, tensor_ranks, solver='fixedpoint', verbose=False, tol=1e-4)
plt.plot(range(len(metric)),metric)
plt.ylabel('L1-Tucker metric')
plt.xlabel('Iteration index')
plt.show()

core,factors,metric=l1hooi(tensor, tensor_ranks, solver='bitflipping', verbose=False, tol=1e-4)
plt.plot(range(len(metric)),metric)
plt.ylabel('L1-Tucker metric')
plt.xlabel('Iteration index')
plt.show()

core,factors,metric=l1hooi(tensor, tensor_ranks, solver='exact', verbose=False, tol=1e-4)
plt.plot(range(len(metric)),metric)
plt.ylabel('L1-Tucker metric')
plt.xlabel('Iteration index')
plt.show()