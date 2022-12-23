import torch

def logt(t, x):
    assert t >= 0
    #assert x > 0 
    if t == 1:
        return torch.log(x)
    else:
        tmp_1 = 1/(1-t)
        tmp_2 = torch.pow(x, 1-t) - 1
        return tmp_1 * tmp_2