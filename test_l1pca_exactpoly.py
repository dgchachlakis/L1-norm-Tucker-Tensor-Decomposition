from l1pca_exactpoly import *
import matplotlib.pyplot as plt

D=5
N=5
matrix_rank=4;
matrix=np.random.randn(D,N)

Qopt, Bopt = l1pca_exactpoly(matrix, matrix_rank, half_sphere=True, verbose=True)
print(Qopt.T@Qopt)