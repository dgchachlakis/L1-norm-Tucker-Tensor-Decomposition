import tensorly as tl
import numpy as np
def metricL1tucker(tensor, factors):
    core = tl.tenalg.multi_mode_dot(tensor, factors, modes = list(range(tensor.ndim)), transpose = True)
    return np.sum(np.abs(core.flatten()))