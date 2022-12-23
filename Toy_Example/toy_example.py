import numpy as np
import matplotlib.pyplot as plt
from utils import tlog,tgaussian
from matplotlib.legend_handler import HandlerTuple
import matplotlib.pylab as pl
def mixture_pdf(x,frac_c,mus,sigmas):
    return np.sum(np.vstack([frac_c[i] * tgaussian(1.0, x, mus[i], sigmas[i]) for i in range(0,len(frac_c))]),axis=0)

def tilt(x,t):
    x=x**(1/t)
    return(x/(np.sum(x)*20/10000))

def integral(m,px_phi,q,grid,MC_size):
    '''Compute integral using MC approx.'''
    q=q/np.sum(q) #normalize
    n=px_phi.shape[1]
    smpld_id = np.random.choice(grid.shape[0], MC_size*(m - 1), True, q)  # Sample the params
    px_phi_smpld= np.asarray([px_phi[smpld_id, i] for i in range(n)])
    c = np.sum(px_phi_smpld.reshape(n, m-1, MC_size), axis=1)
    c = np.repeat(c[np.newaxis,:, :], px_phi.shape[0], axis=0)
    res=m*np.sum(np.sum(tlog(t,(c+px_phi[:,:,np.newaxis])),axis=1),axis=1)/MC_size
    return res

''' Parameters about the free-energy'''
m=1 # multi sample parameter
t = 1  # log-t parameter
beta = 0.1  # beta parameter

print('m: '+str(m)+' t: '+str(t)+' beta: '+str(beta))
''' Empirical Ensemble Risk Minimization (m= finite, n= finite)'''
sigma_lk=0.25
mus = np.asarray([0.5, 0.8, 0.2])  # means
sigmas = np.asarray([0.05, 0.02, 0.02])  # variances
n_tr = 10
colors = pl.cm.coolwarm(np.linspace(0.3, 1, 4))
np.random.seed(1)
MC_rep = 1
BETA_C = False
grid = np.linspace(-10, 1, 1000)
pr=tgaussian(1.0, grid, -5, 5) #
pr = pr / np.sum(pr)
q = np.ones(len(grid)) / len(grid)  # Variational distr.
alpha = 0.975
TV_arr = []  # TV store
TV_arr_clean = []  # TV store
TV = []  # TV store
TV_clean = []  # TV store
plots = []
lgds = []
j = 0
temp_tv = []
temp_tv_clean = []
data = np.load('results/datapoints.npy')
px_phi = np.vstack([tgaussian(1.0, data, mu, sigma_lk) for mu in grid])+10e-20  # Likelihood
losses=-tlog(t,px_phi)
mu_opt = grid[np.argmin(np.mean(-tlog(t,px_phi), axis=1))]
x = np.linspace(-10, 1, 1000)
px_x = tgaussian(1.0, x, mu_opt, sigma_lk)
py = np.transpose(px_x)
py = py / (np.sum(py))
np.save('results/freq'+str(t), py)
q = np.ones(len(grid)) / len(grid)  # Variational distr.
q=q*0
q[np.argmin(np.mean(-tlog(t,px_phi), axis=1))]=1
np.save('results/posterior_freq' + str(t), q)
q = np.ones(len(grid)) / len(grid)  # Variational distr.
for i in range(0, 100):
    if(m==1):
        delta_q = pr * np.exp(beta*np.sum(tlog(t,px_phi),axis=1))
        delta_q =  delta_q / (np.sum(delta_q))
    else:
        int=integral(m,px_phi,q,grid,500)
        delta_q = pr * np.exp(beta*int)
        delta_q =  delta_q / (np.sum(delta_q))
    q = alpha * q + (1 - alpha) * delta_q
    x = np.linspace(-10, 1, 1000)
    plt.clf()
    px_x = np.vstack([tgaussian(1.0, x, mu, sigma_lk) for mu in grid])
    py = (np.transpose(px_x)).dot(q)
    plt.plot(x, py,label='Predictive Dist.',color='tab:green')
    plt.plot(grid,q,label='Posterior Dist.',color='tab:blue')
    plt.scatter(data, -np.ones(len(data)) * 0.001, color='black', marker='+', label='Data Points')
    plt.title('Iteration: '+str(i))
    plt.grid()
    plt.legend()
    plt.pause(0.001)
x = np.linspace(-10, 1, 1000)
px_x = np.vstack([tgaussian(1.0, x, mu, sigma_lk) for mu in grid])
py = (np.transpose(px_x)).dot(q)
py = py / (np.sum(py) )
np.save('results/m=' + str(m) + 't=' + str(t) + 'beta=' + str(beta), py)
np.save('results/posterior_m=' + str(m) + 't=' + str(t) + 'beta=' + str(beta), q)
