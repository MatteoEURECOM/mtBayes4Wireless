import torch
import numpy as np

def normal_prob(x,mu, sigma):
     exponent=(torch.sum(torch.pow(x - mu, 2),axis=1)).type(torch.DoubleTensor)/(2*(sigma**2))
     ret= torch.exp(-exponent)/torch.sqrt(sigma*2*(np.pi)**x.shape[1])
     return ret

def log_normal_prob(x,mu, sigma):
     exponent=-(torch.mean(torch.pow(x - mu, 2),axis=1)).type(torch.DoubleTensor)/(2*sigma**2)
     return exponent