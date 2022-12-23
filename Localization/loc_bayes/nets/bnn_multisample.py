import torch
import torch.nn as nn
from torch.nn import functional as F
import torchbnn as bnn
import tensorflow_probability as tfp
tfd = tfp.distributions

no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")

class BNN(nn.Module):
    def __init__(self, num_neurons_hidden=10,in_shape=71):
        super(BNN, self).__init__()
        self.fc1 =  bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,in_features=in_shape, out_features=num_neurons_hidden)
        self.fc2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,in_features=num_neurons_hidden, out_features=int(num_neurons_hidden/2))
        self.fc3 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,in_features=int(num_neurons_hidden/2), out_features=2)
        self.activ = nn.ELU()
    def forward(self, x_input, m):
        prob_list = []
        for ind_m in range(m):
            x = torch.reshape(x_input, (x_input.shape[0], -1))
            hid1 = self.activ(self.fc1(x))
            hid2 = self.activ(self.fc2(hid1))
            out = self.fc3(hid2)
            prob_list.append(out)
        return prob_list


