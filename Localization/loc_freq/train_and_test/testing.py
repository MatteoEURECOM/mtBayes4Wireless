import torch
from funcs.log_t import logt
from funcs.kl_div import kl
from funcs.gaussian import normal_prob,log_normal_prob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.pylab as pl
import matplotlib.colors as colors
from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal
import geopy.distance
import seaborn as sns
import pandas as pd
import utm
import matplotlib.colors as mcolors
from pyproj import Proj, transform
# one epoch

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def test(args, bnn, test_loader):
    nll = 0
    mse = 0
    nllt = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(args.device), target.to(args.device)
            probs = bnn(data, args.m_te)
            log_p_x = torch.stack([log_normal_prob(target, p, args.sigma) for p in probs])
            p_x = torch.exp(log_p_x)
            if (args.m == 1):
                log_t_avg_prob = logt(args.t, torch.mean(torch.exp(log_p_x), 0) + 10e-300)
            else:
                a = torch.max(log_p_x, 0)[0].detach()
                a_exp = torch.max(p_x, 0)[0].detach() + 10e-300
                log_t_avg_prob = (1 + (1 - args.t) * logt(args.t, a_exp)) * logt(args.t, torch.mean(torch.exp(log_p_x - a), 0)) + logt(args.t, a_exp)

            log_avg_prob = torch.logsumexp(torch.add(log_p_x, -np.log(args.m_te)), axis=0)
            nll += torch.mean(log_avg_prob)
            nllt += torch.mean(log_t_avg_prob)
            mse += torch.sum(torch.mean(torch.stack([torch.sqrt(torch.sum(torch.pow(p - target, 2),axis=1)) for p in probs]),0))
    test_nll = nll/len(test_loader.dataset)
    test_nllt = nllt / len(test_loader.dataset)
    test_mse = mse/len(test_loader.dataset)
    return test_nll.cpu().data.numpy(),test_nllt.cpu().data.numpy(),test_mse.cpu().data.numpy()

def kde_estimation(test_loader):
    data=test_loader.dataset
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    g = np.meshgrid(x, y)
    positions = np.append(g[0].reshape(-1, 1), g[1].reshape(-1, 1), axis=1)
    RSSI=data.RSSI
    loc=data.loc
    sigma=1
    for j in range(100,120):
        print(j)
        exponent = (np.sum((RSSI - RSSI[j,:])**2, axis=1)) / (2 * (sigma ** 2))
        scores = np.exp(-exponent) / (sigma  * 2 * np.pi)
        t=np.zeros((100,100))
        for i in range(0,RSSI.shape[0]):
            var = multivariate_normal(mean=loc[i, :], cov=[[0.001, 0], [0,0.001]])
            t=t+scores[i]*np.reshape(var.pdf(positions),(100,100))
        plt.contourf(g[0], g[1], t)
        plt.show()
    return 0


def mse_avg(args, bnn, test_loader,METER):
    MSE = []
    inProj = Proj(init='epsg:3857')
    outProj = Proj(init='epsg:4326')
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(args.device), target.to(args.device)
        probs = bnn(data, args.m_te)  # m: number of multisamples
        stacked = np.mean(torch.stack(probs).detach().numpy(),axis=0)
        target = target.detach().numpy()
        MSE.append(np.sqrt(np.sum((stacked - target) ** 2, axis=1)))
        if(METER):
            target = test_loader.dataset.label.inverse_transform(target)
            stacked = test_loader.dataset.label.inverse_transform(stacked)
            if (test_loader.dataset.dataset_name == 'UTS.csv'):
                MSE.append(np.sqrt(np.sum((stacked - target) ** 2, axis=1)))
            elif (test_loader.dataset.dataset_name == 'UJI.csv'):
                MSE.append(np.asarray([geopy.distance.geodesic(transform(inProj, outProj, target[ind,0], target[ind,1]), transform(inProj, outProj, stacked[ind, 0], stacked[ind, 1]) ,ellipsoid='WGS-84').m for ind in range(0, target.shape[0])]))
            else:
                MSE.append(np.asarray([geopy.distance.geodesic(target[i, :], stacked[i, :]).m for i in range(0, target.shape[0])]))
        else:
            MSE.append(np.sqrt(np.sum((stacked - target) ** 2, axis=1)))
    MSE=np.hstack(MSE)
    return np.mean(MSE),np.std(MSE)/MSE.shape[0]
