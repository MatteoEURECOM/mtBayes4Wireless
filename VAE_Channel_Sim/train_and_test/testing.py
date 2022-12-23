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
import matplotlib.colors as mcolors
import tensorflow as tf
import sklearn
import scikitplot as skplt
import copy


def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    #plt.show()

def compute_mmd(args,bnn,test_loader):
    plt.rcParams.update({"text.usetex": True,"font.family": "serif","font.serif": ["Computer Modern Roman"],'font.size': 16,'lines.linewidth':2})
    to_generate=len(test_loader.dataset)
    plt.close()
    ch_VAE=np.vstack(bnn.generate_channels(to_generate,1))
    test_data=test_loader.dataset.X
    score=maximum_mean_discrepancy(ch_VAE, test_data, kernel=sklearn.metrics.pairwise.rbf_kernel)
    if False:
        X = np.concatenate([np.linspace(0, 1, 128) for i in range(0, ch_VAE.shape[0])], axis=0)
        Y = np.hstack(ch_VAE)
        heatmap, xedges, yedges = np.histogram2d(X, Y, range=[[0, 0.6], [0, 15]], bins=[128*6, 200], normed=False)
        sigma=args.sigma*200/8
        pdf = gaussian_filter(heatmap, sigma=[1,sigma])
        pdf = pdf.T / pdf.max(axis=1)
        my_cmap = copy.copy(plt.cm.get_cmap('binary'))  # get a copy of the gray color map
        my_cmap.set_bad(alpha=0)  # set how the colormap handles 'bad' values
        fig = plt.figure(figsize=(6, 4))
        plt.imshow(np.flipud(pdf), origin='upper', extent=[0,60,0,15],cmap=my_cmap, aspect='auto')
        plt.grid(alpha=0.5)
        plt.subplots_adjust(bottom=0.2)
        plt.ylabel(r'$x_i$')
        plt.xlabel(r'$i$')
        #plt.annotate(r'$\xi(x)$ components', ha='center', va='bottom', xytext=(40, 5), xy=(25, 1),   arrowprops={'facecolor': 'black', 'width': 0.3, 'headwidth': 4, 'headlength': 5, 'connectionstyle': 'arc3,rad=.1'})
        #plt.annotate(r'$\xi(x)$ components', ha='center', va='bottom', xytext=(40, 5), xy=(35, 1),  arrowprops={'facecolor': 'black', 'width': 0.3, 'headwidth': 4, 'headlength': 5, 'connectionstyle': 'arc3,rad=.1'})
        plt.savefig('Density_t'+str(args.t)+'_new.pdf')
        plt.show()
        X = np.concatenate([np.linspace(0, 1, 128) for i in range(0, test_data.shape[0])], axis=0)
        Y = np.hstack(test_data)
        heatmap, xedges, yedges = np.histogram2d(X, Y, range=[[0, 0.6], [0, 15]], bins=[128 * 6, 200], normed=False)
        sigma = args.sigma * 200 / 8
        pdf = gaussian_filter(heatmap, sigma=[1, sigma])
        pdf = pdf.T / pdf.max(axis=1)
        my_cmap = copy.copy(plt.cm.get_cmap('binary'))  # get a copy of the gray color map
        my_cmap.set_bad(alpha=0)  # set how the colormap handles 'bad' values
        fig = plt.figure(figsize=(6, 4))
        plt.imshow(np.flipud(pdf), origin='upper', extent=[0,60, 0, 15], cmap=my_cmap, aspect='auto')
        plt.grid(alpha=0.5)
        plt.subplots_adjust(bottom=0.2)
        plt.ylabel(r'$x_i$')
        plt.xlabel(r'$i$')
        plt.savefig('Density_ID.pdf')
        plt.show()
    return score
def test_latent_space(args, bnn, test_loader):
    z = []
    mus = []
    log_vars = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(args.device)
            latent, mu, log_var = bnn.get_latent(data)
            z.append(latent.detach().numpy())
            mus.append(mu.numpy())
            log_vars.append(log_var.numpy())
        z=np.vstack(z)
        mus=np.vstack(mus)
        log_vars=np.vstack(log_vars)
        for i in range(0,5):
            plt.hist(mus[:,i], bins='auto',density=True)
            x = np.arange(-5, 5, .01)
            mean=0
            variance=1
            f = np.exp(-np.square(x - mean) / 2 * variance) / (np.sqrt(2 * np.pi * variance))
            plt.plot(x,f)
            #plt.show()
        return 0

def auroc(args, bnn, test_loader):
    scores = []
    with torch.no_grad():
        for i in range(0,int(len(test_loader.dataset.X)/100)):
            data=torch.tensor(test_loader.dataset.X[i*100:(i+1)*100,:])
            data = data.to(args.device)
            probs,mus,log_vars = bnn(data,args.m_te)
            if(not args.BayesianDec):
                probs = torch.vstack([torch.mean((p - data) ** 2, axis=1) for p in probs]).numpy()
                scores.append(np.mean(probs,axis=0))
            else:
                probs=torch.vstack([torch.exp(-torch.mean((p-data)**2,axis=1)/(2*args.sigma**2)) for p in probs]).numpy()
                probs=np.mean(probs,axis=0)
                scores.append(probs)
    if (not args.BayesianDec):
        scores=np.hstack(scores)
        scores=-scores/np.max(scores)
    else:
        scores=np.hstack(scores)/args.m_te
    scores[np.isnan(scores)]=1
    scores[scores==np.inf] = 0
    labels=test_loader.dataset.OOD
    fper, tper, thresholds =  sklearn.metrics.roc_curve(labels, scores)
    plot_roc_curve(fper, tper)
    print(sklearn.metrics.roc_auc_score(labels, scores))
    return fper, tper, sklearn.metrics.roc_auc_score(labels, scores)



def test(args, bnn, test_loader):
    nll = 0
    mse = 0
    nllt = 0
    with torch.no_grad():
        for i in range(0, int(len(test_loader.dataset.X) / 100)):
            data = torch.tensor(test_loader.dataset.X[i * 100:(i + 1) * 100, :])
            data= data.to(args.device)
            probs,mus,log_vars = bnn(data,args.m_te)
            log_p_x = torch.stack([log_normal_prob(data, p, args.sigma) for p in probs])
            log_t_avg_prob = logt(args.t, torch.sum(torch.exp(log_p_x-np.log(args.m_te)), 0))
            log_avg_prob = torch.logsumexp(torch.add(log_p_x, -np.log(args.m_te)), axis=0)
            nll += torch.mean(log_avg_prob)
            nllt += torch.mean(log_t_avg_prob)
            mse += torch.sum(torch.mean(torch.stack([torch.sqrt(torch.sum(torch.pow(p - data, 2), axis=1)) for p in probs]), 0))
        test_nll = nll / len(test_loader.dataset)
        test_nllt = nllt / len(test_loader.dataset)
        test_mse = mse / len(test_loader.dataset)
        return test_nll.cpu().data.numpy(), test_nllt.cpu().data.numpy(), test_mse.cpu().data.numpy()

def maximum_mean_discrepancy(x, y, kernel=sklearn.metrics.pairwise.rbf_kernel ):
    r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.

    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
    the distributions of x and y. Here we use the kernel two sample estimate
    using the empirical mean of the two distributions.

    MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
              = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },

    where K = <\phi(x), \phi(y)>,
    is the desired kernel function, in this case a radial basis kernel.

    Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      kernel: a function which computes the kernel in MMD. Defaults to the
              GaussianKernelMatrix.

    Returns:
      a scalar denoting the squared maximum mean discrepancy loss.
    """
    gamma=0.1
    cost = np.mean(kernel(x, x,gamma=gamma))
    cost += np.mean(kernel(y, y,gamma=gamma))
    cost -= 2 * np.mean(kernel(x, y,gamma=gamma))
    # We do not allow the loss to become negative.
    print('MMD:')
    print(cost)
    return cost