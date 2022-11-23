import numpy as np
import torch
def weight_variable(name, shape, pad=True):
    initial = np.random.uniform(-0.01, 0.01, size=shape)
    if pad == True:
        initial[0] = np.zeros(shape[1])
    result=torch.tensor(initial)
    return result

v_attribute=weight_variable('v_attribute',(48,300))