from l1hosvd import *

tensor=np.random.randn(15,15,15,15)
tensor_ranks=np.array([5,6,7,5])
core,factors=l1hosvd(tensor,tensor_ranks)

