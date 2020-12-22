from l1hosvd import *
print(50*'\n')
tensor=np.random.randn(3,3,3)
tensor_ranks=np.array([2,2,2])


core_fp,factors_fp=l1hosvd(tensor,tensor_ranks,solver='fixed-point',verbose=True,tol=1e-4)
core_bf,factors_bf=l1hosvd(tensor,tensor_ranks,solver='bit-flipping',verbose=True,tol=1e-4)
core,factors=l1hosvd(tensor,tensor_ranks,solver='exact-poly',verbose=True,tol=1e-4)


