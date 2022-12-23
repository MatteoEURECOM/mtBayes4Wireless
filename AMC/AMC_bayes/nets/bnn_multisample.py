import torch
import torch.nn as nn
from torch.nn import functional as F
import torchbnn as bnn
import tensorflow_probability as tfp
import math
tfd = tfp.distributions


no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")

class BNN(nn.Module):
    def __init__(self):
        super(BNN,self).__init__()
        prior_sigma=1.
        self.cnn1 = bnn.BayesConv2d(prior_mu=0, prior_sigma=.1,in_channels=1, out_channels=16, kernel_size=(2,3), stride=(1,2), padding=0, groups=1, bias=True, padding_mode='zeros')
        self.cnn2 = bnn.BayesConv2d(prior_mu=0, prior_sigma=.1, in_channels=16, out_channels=2, kernel_size=(1,2), stride=2, padding=0, groups=1, bias=True, padding_mode='zeros')
        self.cnn1.prior_sigma = prior_sigma
        self.cnn1.prior_log_sigma = math.log(prior_sigma)
        self.cnn2.prior_sigma = prior_sigma
        self.cnn2.prior_log_sigma = math.log(prior_sigma)
        self.fc1 =  bnn.BayesLinear(prior_mu=0, prior_sigma=.1,in_features=64, out_features=30)
        self.fc2 = bnn.BayesLinear(prior_mu=0, prior_sigma=.1,in_features=30, out_features=8)
        self.fc1.prior_sigma = prior_sigma
        self.fc1.prior_log_sigma = math.log(prior_sigma)
        self.fc2.prior_sigma = prior_sigma
        self.fc2.prior_log_sigma = math.log(prior_sigma)
        self.activ = nn.ELU()
        self.out_activ= nn.Softmax(dim=1)
    def forward(self, x_input, m):
        prob_list = []
        for ind_m in range(m):
            x = torch.reshape(x_input, (x_input.shape[0], 1,2,-1))
            x_pad=torch.nn.functional.pad(x,(2,2))
            hid1 = self.activ(self.cnn1(x_pad))
            hid2 = self.activ(self.cnn2(hid1))
            hid3 = self.activ(self.fc1(torch.reshape(hid2,(x_input.shape[0],-1))))
            out = self.out_activ(self.fc2(hid3))
            prob_list.append(out)
        return sum(prob_list)/m
