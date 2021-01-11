import numpy as np
def metricL1pca(X, Q):
    return np.sum(np.abs((X.T @ Q).flatten()))