import numpy as np
import matplotlib.pyplot as plt
#from utils import tlog,tgaussian
import matplotlib.pylab as pl
from matplotlib.legend_handler import HandlerTuple

'''             SINGLE PLOT                 '''

colors = pl.cm.coolwarm(np.linspace(0.3,1,3))
plt.rcParams.update({"text.usetex": True,"font.family": "sans-serif","font.sans-serif": ["Computer Modern Sans Serif"],'font.size': 14,'lines.linewidth':2})
plt.rc('font', family='serif', serif='Computer Modern Roman', size=13)

clean=np.load('results/clean.npy')
data=np.load('results/datapoints.npy')
test=np.load('results/m=1t='+str(1)+'beta=1.npy')
HORIZONTAL=True
if HORIZONTAL:
    fig, axs = plt.subplots(1,2, sharex=True, figsize=(12,3.5))
    plt.subplots_adjust(bottom=0.1)
else:
    fig, axs = plt.subplots(2, sharex=True, figsize=(6, 6.5))
fig.tight_layout()
i=0
m=10
lgds=[]
plots=[]
plt_s, =axs[0].plot(np.linspace(-15, 1, 5000), clean, color='green', linestyle='--', label=r'$\nu(x)$')
plots.append(plt_s)
lgds.append(r'Target distribution')
p1 = axs[0].scatter(data, -np.ones(len(data)) * 0.075, color='black', marker='x', label=r'$\mathcal{D}$')
p2 = axs[0].scatter(np.min(data), -0.075, color='red', marker='x')
plt_s, =axs[0].plot(np.linspace(-10, 1, 1000),(999./11)*np.load('results/m='+str(m)+'t='+str(1)+'beta=1.npy')/np.sum(np.load('results/m='+str(m)+'t='+str(1)+'beta=1.npy')),label=r'$\mathcal{J}^{'+str(m)+'}$',color='tab:red')
plots.append(plt_s)
lgds.append(r'$('+ str(m) + ',' + str(1) +r' )$-Robust Bayesian')
plt_s, = axs[0].plot(np.linspace(-10, 1, 1000),(999./11)*np.load('results/m=1t=0.4beta=1.npy'), color='tab:grey', label=r'$\mathcal{J}$')
plots.append(plt_s)
lgds.append(r'$(1,0.4)$-Robust Bayesian')

for t in [0.4]:
    plt_s, = axs[0].plot(np.linspace(-10, 1, 1000),(999./11)*np.load('results/m='+str(m)+'t='+str(t)+'beta=1.npy'),label=r'$\mathcal{J}^{'+str(m)+'}_{'+str(t)+'}$',color='tab:blue')
    plots.append(plt_s)
    lgds.append(r'$(' + str(m) + ',' + str(t) + r')$-Robust Bayesian ')
    i=i+1
axs[0].legend([p1, p2,plots[0],plots[1],plots[2],plots[3]], [r'Target distribution samples',r'Outlier',lgds[0],lgds[1],lgds[2],lgds[3]], handler_map={tuple: HandlerTuple(ndivide=None)})
axs[0].set_xlim([-9, 0])
axs[0].set_ylim([-0.13,2])
axs[0].grid()


i=0
lgds=[]
plots=[]
p1 = axs[1].scatter(data, -np.ones(len(data)) * 0.15, color='black', marker='x', label=r'$\mathcal{D}$')
p2 = axs[1].scatter(np.min(data), -0.15, color='red', marker='x')
plt_s, =axs[1].plot(np.linspace(-10, 1, 1000),(999./11)*np.load('results/posterior_m='+str(m)+'t=1beta=1.npy')/np.sum(np.load('results/posterior_m='+str(m)+'t=1beta=1.npy')),label=r'$\mathcal{J}^{'+str(m)+'}$',color='tab:red')
plots.append(plt_s)
lgds.append(r'$('+ str(m) + ',' + str(1) +r' )$-Robust Bayesian ')
plt_s, =axs[1].plot(np.linspace(-10, 1, 1000),(999./11)*np.load('results/posterior_m=1t=0.4beta=1.npy')/np.sum(np.load('results/posterior_m=1t=0.4beta=1.npy')), color='tab:grey', label=r'$\mathcal{J}$')
plots.append(plt_s)
lgds.append(r'$(1,0.4)$-Robust Bayesian')

for t in [0.4]:
    plt_s, =axs[1].plot(np.linspace(-10, 1, 1000),(999./11)*np.load('results/posterior_m='+str(m)+'t='+str(t)+'beta=1.npy')/np.sum(np.load('results/posterior_m='+str(m)+'t='+str(t)+'beta=1.npy')),label=r'$\mathcal{J}^{'+str(m)+'}_{'+str(t)+'}$',color='tab:blue')
    plots.append(plt_s)
    lgds.append(r'$(' + str(m) + ',' + str(t) + r')$-Robust Bayesian')
    i=i+1
axs[1].legend([p1, p2,plots[0],plots[1],plots[2]], [r'Target distribution samples',r'Outlier',lgds[0],lgds[1],lgds[2]], handler_map={tuple: HandlerTuple(ndivide=None)},loc='upper left')
axs[0].set_xlabel(r'Channel Gain [dB]')
axs[1].set_xlabel(r'$\theta$')
axs[1].set_xlim([-9, 0])
axs[1].set_ylim([-0.3,3])
axs[1].grid()
axs[1].set_ylabel(r'$q(\theta)$')
axs[0].set_ylabel(r'$p(x|\theta_{freq})$ and $p(x|q)$')
axs[1].set_xticks([-8,-6,-4,-2,0])
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.1)

