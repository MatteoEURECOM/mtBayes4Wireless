import torch
import torch.nn as nn
import torchbnn as bnn
import tensorflow_probability as tfp
import math
tfd = tfp.distributions

no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")

class VAE(nn.Module):
    def __init__(self,in_size,BayesEnc=False,BayesianDec=False,dim=10):
        super(VAE, self).__init__()
        self.dim=dim
        hid_dim=20
        if(not BayesEnc):
            self.fc1e = torch.nn.Linear(in_features=in_size,out_features=hid_dim)
            self.fc2e = torch.nn.Linear(in_features=hid_dim, out_features=hid_dim)
            self.fc3e = torch.nn.Linear(in_features=hid_dim, out_features=dim*2)
        else:
            prior_sigma=1
            self.fc1e = bnn.BayesLinear(prior_mu=0, prior_sigma=.1, in_features=in_size, out_features=hid_dim)
            self.fc2e = bnn.BayesLinear(prior_mu=0, prior_sigma=.1, in_features=hid_dim, out_features=hid_dim)
            self.fc3e = bnn.BayesLinear(prior_mu=0, prior_sigma=.1, in_features=hid_dim, out_features=dim*2)
            self.fc1e.prior_sigma = prior_sigma
            self.fc1e.prior_log_sigma = math.log(prior_sigma)
            self.fc2e.prior_sigma = prior_sigma
            self.fc2e.prior_log_sigma = math.log(prior_sigma)
            self.fc3e.prior_sigma = prior_sigma
            self.fc3e.prior_log_sigma = math.log(prior_sigma)
        if (not BayesianDec):
            self.fc1d = torch.nn.Linear(in_features=dim, out_features=hid_dim)
            self.fc2d = torch.nn.Linear(in_features=hid_dim, out_features=hid_dim)
            self.fc3d = torch.nn.Linear(in_features=hid_dim, out_features=in_size)
        else:
            prior_sigma = 1
            self.fc1d = bnn.BayesLinear(prior_mu=0, prior_sigma=.1, in_features=dim, out_features=hid_dim)
            self.fc2d = bnn.BayesLinear(prior_mu=0, prior_sigma=.1, in_features=hid_dim, out_features=hid_dim)
            self.fc3d = bnn.BayesLinear(prior_mu=0, prior_sigma=.1, in_features=hid_dim, out_features=in_size)
            self.fc1d.prior_sigma = prior_sigma
            self.fc1d.prior_log_sigma = math.log(prior_sigma)
            self.fc2d.prior_sigma = prior_sigma
            self.fc2d.prior_log_sigma = math.log(prior_sigma)
            self.fc3d.prior_sigma = prior_sigma
            self.fc3d.prior_log_sigma = math.log(prior_sigma)
        self.activ = nn.ELU()
        self.ReLU = nn.ReLU()
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample


    def forward(self, x_input,m):
        prob_list = []
        mu_list=[]
        log_var_list = []
        hid1 = self.activ(self.fc1e(x_input))
        hid2 = self.activ(self.fc2e(hid1))
        hid3 =  self.fc3e(hid2).view(-1, 2,  self.dim)
        mu = hid3[:, 0, :]  # the first feature values as mean
        log_var = hid3[:, 1, :]  # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        for ind_m in range(m):
            hid4 = self.activ(self.fc1d(z))
            hid5 = self.activ(self.fc2d(hid4))
            out = self.ReLU(self.fc3d(hid5))
            prob_list.append(out)
            mu_list.append(mu)
            log_var_list.append(log_var)
        return prob_list,mu_list,log_var_list


    def get_latent(self, x_input):
        hid1 = self.activ(self.fc1e(x_input))
        hid2 = self.activ(self.fc2e(hid1))
        hid3 =  self.fc3e(hid2).view(-1, 2,  self.dim)
        mu = hid3[:, 0, :]  # the first feature values as mean
        log_var = hid3[:, 1, :]  # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        return z,mu,log_var

    def generate_channels(self, batch_size,m):
        prob_list = []
        mu=torch.tensor(0).repeat(batch_size,self.dim)
        log_var=torch.tensor(0).repeat(batch_size,self.dim)
        for ind_m in range(m):
            z = self.reparameterize(mu, log_var)
            hid4 = self.activ(self.fc1d(z))
            hid5 = self.activ(self.fc2d(hid4))
            out = self.ReLU(self.fc3d(hid5))
            prob_list.append(out.detach().numpy())
        return prob_list

