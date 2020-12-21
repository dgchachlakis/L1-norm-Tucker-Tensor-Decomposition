from l1pca_fixedpoint import *
import matplotlib.pyplot as plt

D=50
N=100
matrix_rank=40;
matrix=np.random.randn(D,N)

# Test l1pca_fixed point without initialization
Q, metric_evolution=l1pca_fixedpoint(matrix,matrix_rank,verbose=True)
plt.plot(range(len(metric_evolution)),metric_evolution)
plt.ylabel('L1-PCA metric')
plt.xlabel('Iteration index')
plt.show()

# Test l1pca_fixed  with initialization
Qin=linalg.orth(np.random.randn(D,matrix_rank))
Q2, metric_evolution2=l1pca_fixedpoint(matrix,matrix_rank,Qin,verbose=True)
plt.plot(range(len(metric_evolution2)),metric_evolution2)
plt.ylabel('L1-PCA metric')
plt.xlabel('Iteration index')
plt.show()