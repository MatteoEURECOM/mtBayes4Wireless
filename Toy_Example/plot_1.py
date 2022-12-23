import numpy as np
import matplotlib.pyplot as plt
#from utils import tlog,tgaussian
import matplotlib.pylab as pl
from matplotlib.legend_handler import HandlerTuple
#Latex Font

i=0
m=1
colors = pl.cm.coolwarm(np.linspace(0.3,1,2))


'''             SINGLE PLOT                 '''
HORIZONTAL=True
pl.cm.coolwarm(np.linspace(0.3,1,2))
plt.rcParams.update({"text.usetex": True,"font.family": "sans-serif","font.sans-serif": ["Computer Modern Sans Serif"],'font.size': 14,'lines.linewidth':2})
plt.rc('font', family='serif', serif='Computer Modern Roman', size=13)
if HORIZONTAL:
    fig, axs = plt.subplots(1,2, sharex=True, figsize=(12,3.5))
    plt.subplots_adjust(bottom=0.1)
else:
    fig, axs = plt.subplots(2, sharex=True, figsize=(6, 6.5))
fig.tight_layout()
clean=np.load('results/clean.npy')
data=np.load('results/clean_datapoints.npy')
test=np.load('results/m=1t=1beta=1.npy')
i=0
m=1
lgds=[]
plots=[]
plt_s, =axs[0].plot(np.linspace(-15, 1, 5000), clean, color='green', linestyle='--', label=r'$\nu(x)$')
plots.append(plt_s)
lgds.append(r'Target distribution')
plt_s, =axs[0].plot(np.linspace(-10, 1, 1000),(999./11)*np.load('results/freq1.npy'),label=r'$\hat{\mathcal{R}}$',linestyle='-.',color='black')
plots.append(plt_s)
lgds.append(r'Frequentist')
p1 = axs[0].scatter(data, -np.ones(len(data)) * 0.075, color='black', marker='x', label=r'$\mathcal{D}$')
p2 = axs[0].scatter(np.min(data), -0.075, color='black', marker='x')
beta=1
plt_s, = axs[0].plot(np.linspace(-10, 1, 1000),(999./11)*np.load('results/m='+str(m)+'t=1beta='+str(beta)+'.npy'), color='tab:blue', label=r'$\mathcal{J}$')
plots.append(plt_s)
lgds.append(r'Bayesian $\beta=' + str(beta) + '$')

for beta in [0.1]:
    plt_s, = axs[0].plot(np.linspace(-10, 1, 1000),(999./11)*np.load('results/m='+str(m)+'t=1beta='+str(beta)+'.npy'),color=colors[i])
    plots.append(plt_s)
    lgds.append(r'Bayesian $\beta=' + str(beta) + '$')
    i=i+1
beta=1
plt_s, = axs[0].plot(np.linspace(-10, 1, 1000),(999./11)*np.load('results/m=10t=1beta='+str(beta)+'.npy'), color='tab:red', label=r'$\mathcal{J}$')
plots.append(plt_s)
lgds.append(r'$(10,1)$-Robust Bayesian')
axs[0].legend([p1,plots[0],plots[1],plots[2],plots[3],plots[4]], [r'Target distribution samples',lgds[0],lgds[1],lgds[2],lgds[3],lgds[4]], handler_map={tuple: HandlerTuple(ndivide=None)})
axs[0].set_xlim([-7, 0])
axs[0].set_ylim([-0.13,2.5])
axs[0].grid()


i=0
lgds=[]
plots=[]
p1 = axs[1].scatter(data, -np.ones(len(data)) * 0.3, color='black', marker='x', label=r'$\mathcal{D}$')
p2 = axs[1].scatter(np.min(data), -0.3, color='black', marker='x')
plt_s, =axs[1].plot(np.argmax(np.load('results/posterior_freq1.npy'))*(11/999)-10,1, linestyle = 'None',marker='o',label=r'$\hat{\mathcal{R}}$',color='black')
plots.append(plt_s)
lgds.append(r'Frequentist')
plt.vlines(np.argmax(np.load('results/posterior_freq1.npy'))*(11/999)-10,0,1,color='black')
plt_s, =axs[1].plot(np.linspace(-10, 1, 1000),(999./11)*np.load('results/posterior_m=1t=1beta=1.npy')/np.sum(np.load('results/posterior_m=1t=1beta=1.npy')), color='tab:blue', label=r'$\mathcal{J}$')
plots.append(plt_s)
lgds.append(r'Bayesian $\beta=1$')
for beta in [0.1]:
    plt_s, =axs[1].plot(np.linspace(-10, 1, 1000),(999./11)*np.load('results/posterior_m='+str(m)+'t=1beta='+str(beta)+'.npy')/np.sum(np.load('results/posterior_m='+str(m)+'t=1beta='+str(beta)+'.npy')),color=colors[i])
    plots.append(plt_s)
    lgds.append(r'Bayesian $\beta=' + str(beta) + '$')
    i=i+1
beta=1
ciao=np.load('results/posterior_m=10t=1beta='+str(beta)+'.npy')
plt_s, = axs[1].plot(np.linspace(-10, 1, 1000),(999./11)*ciao, color='tab:red', label=r'$\mathcal{J}$')
plots.append(plt_s)
lgds.append(r'$(10,1)$-Robust Bayesian')
axs[1].legend([p1,plots[0],plots[1],plots[2],plots[3]], [r'Target distribution samples',lgds[0],lgds[1],lgds[2],lgds[3]], handler_map={tuple: HandlerTuple(ndivide=None)},loc='upper left')
axs[0].set_xlabel(r'Channel Gain [dB]')
axs[1].set_xlabel(r'$\theta$')
axs[1].set_ylabel(r'$q(\theta)$')
axs[0].set_ylabel(r'$p(x|\theta_{freq})$ and $p(x|q)$')
axs[1].set_xlim([-7, 0])
axs[1].set_ylim([-0.6,11])
axs[1].grid()
axs[1].set_xticks([-6,-4,-2,0])
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.1)
if HORIZONTAL:
    plt.savefig('Fig2a.pdf')
    plt.show()

