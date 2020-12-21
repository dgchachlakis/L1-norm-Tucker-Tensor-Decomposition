from l1hosvd import *
print(50*'\n')
tensor=np.random.randn(5,5,5,5)
tensor_ranks=np.array([2,3,2,4])


core_fp,factors_fp=l1hosvd(tensor,tensor_ranks,solver='fixedpoint',verbose=True,tol=1e-4)
core_bf,factors_bf=l1hosvd(tensor,tensor_ranks,solver='bitflipping',verbose=True,tol=1e-4)
core,factors=l1hosvd(tensor,tensor_ranks,solver='exact_exhaustive',verbose=True,tol=1e-4)


