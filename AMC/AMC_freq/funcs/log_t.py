import torch

def logt(t, x):
    assert t >= 0
    if t == 1:
        return torch.log(x+10e-10)
    else:
        logt = (1/(1-t))*(torch.pow(x+10e-10, 1-t) - 1)
        return logt
