import numpy as np
import algorithms as alg
import matplotlib.pyplot as plt
from scipy import linalg
from utils import * 
tensor = np.random.randn(3, 3, 3)
tensor_ranks = (2, 2, 2)
factors = []
for n in range(3):
    factors.append(linalg.orth(np.random.randn(tensor.shape[n], tensor_ranks[n])))
# Example 1:
# ====L1-HOOI and L1-HOSVD with fixed-point underlying L1-PCA solver
core1, factors1, metricEvolution1 = alg.l1hooi(tensor, tensor_ranks, factors, solver = 'fixedpoint')
core1b, factors1b = alg.l1hosvd(tensor, tensor_ranks, factors, solver = 'fixedpoint')
met1 = metricL1tucker(tensor, factors1b)
# Example 2:
# ====L1-HOOI and L1-HOSVD with bitflipping underlying L1-PCA solver
core2, factors2, metricEvolution2 = alg.l1hooi(tensor, tensor_ranks, factors, solver = 'bitflipping')
core2b, factors2b = alg.l1hosvd(tensor, tensor_ranks, factors, solver = 'bitflipping')
met2 = metricL1tucker(tensor, factors2b)

# Example 3:
# ====L1-HOOI and L1-HOSVD with exactpoly underlying L1-PCA solver
core3, factors3, metricEvolution3 = alg.l1hooi(tensor, tensor_ranks, factors, solver = 'exactpoly')
core3b, factors3b = alg.l1hosvd(tensor, tensor_ranks, factors, solver = 'exactpoly')
met3 = metricL1tucker(tensor, factors3b)

# Example 4:
# ====L1-HOOI and L1-HOSVD with exact underlying L1-PCA solver
core4, factors4, metricEvolution4 = alg.l1hooi(tensor, tensor_ranks, factors, solver = 'exact')
core4b, factors4b = alg.l1hosvd(tensor, tensor_ranks, factors, solver = 'exact')
met4 = metricL1tucker(tensor, factors4b)


plt.figure()
plt.plot(metricEvolution1, '-r', label = "L1-HOOI w/ L1-PCA solver: fixed-point")
plt.plot([0, len(metricEvolution1)], [met1, met1], '--r', label = "L1-HOSVD w/ L1-PCA solver: fixed-point")

plt.plot(metricEvolution2, '-b', label = "L1-HOOI w/ L1-PCA solver: bit-flipping")
plt.plot([0, len(metricEvolution2)], [met2, met2], '--b', label = "L1-HOSVD w/ L1-PCA solver: bit-flipping")

plt.plot(metricEvolution3, '-k', label = "L1-HOOI w/ L1-PCA solver: exact-poly")
plt.plot([0, len(metricEvolution3)], [met3, met3], '--k', label = "L1-HOSVD w/ L1-PCA solver: exact-poly")

plt.plot(metricEvolution4, '-m', label = "L1-HOOI w/ L1-PCA solver: exact")
plt.plot([0, len(metricEvolution4)], [met4, met4], '--m', label = "L1-HOSVD w/ L1-PCA solver: exact")


plt.ylabel('L1-Tucker metric')
plt.xlabel('Iteration index')
plt.legend()
plt.show()