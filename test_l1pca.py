from tests.test_l1pca_solvers import *

#print('Testing exact solver (exhaustive search)...')
#Q,B=test_l1pca_exact()
#print('Success...')

#print('Testing exact solver (polynomial search)...')
#Q,B=test_l1pca_exactpoly()
#print('Success...')

#print('Testing approximate solver (fixed-point)...')
#Q,metric=test_l1pca_fixedpoint()
#print('Success...')

#print('Testing approximate solver (bit-flipping)...')
#Q,metric=test_l1pca_bitflipping()
#print('Success...')

print(2*'\n')
print('Comparing all solvers...')
compare_l1solvers()
print('Success...')