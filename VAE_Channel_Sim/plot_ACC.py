import numpy as np
import matplotlib.pylab as pl
from matplotlib import cm
import matplotlib.pyplot as plt
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def movingaverage(interval, window_size):
    interval=np.concatenate((interval,interval[-window_size+1:]))
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'valid')

plt.rc('font', family='serif', serif='Computer Modern Roman', size=20)
plt.rc('text', usetex=True)
#Table Res
color=['tab:grey','orangered','steelblue']
plt.rcParams.update({'figure.autolayout': True})
ts=[1]
ms=[5]
epss=[0,0.1,0.3]
#Table Res
color=['tab:grey','orangered','steelblue']
plt.rcParams.update({'figure.autolayout': True})
lines=['-.',':','solid']
to_plot=[]
plt.figure(figsize=(5,6))
for k in range(0,len(ts)):
    t=ts[k]
    to_plot = []
    std=[]
    for j in range(0,len(epss)):
        ls = lines[k]
        c = color[k]
        m=ms[0]
        acc = np.mean(np.load('logs/ACC_' + str(t) + '_freq_eps_'+str(epss[j])+'.npy', allow_pickle=True),axis=0)
        std = np.std(np.load('logs/ACC_' + str(t) + '_freq_eps_' + str(epss[j]) + '.npy', allow_pickle=True), axis=0)
        plt.plot(acc[0], acc[1],color=c, marker='o',linewidth=1.5, linestyle=ls,label=r'$m=$ ' + str(m)+ '$t=$ '+ str(t)+'$\epsilon=$' + str(epss[j]))
        #plt.fill_between(acc[0], acc[1] -  1.96*std[1]/np.sqrt(5), acc[1] +  1.96*std[1]/np.sqrt(5))
plt.grid(True,which="both", linestyle='--')
plt.annotate(r'$\mathcal{J}^{5}_1$',ha = 'center', va = 'bottom',xytext = (1.25, 0.35),xy = (-1.62,0.5),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=.1'})
plt.annotate(r'$\mathcal{J}^{5}_{0.6}$',ha = 'center', va = 'bottom',xytext = (-10, 0.5),xy = (-0.54,0.59),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=-.1'})
plt.annotate(r'$\mathcal{J}^{5}_{0.8}$',ha = 'center', va = 'bottom',xytext = (11, 0.38),xy = (6.17,0.525),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=-.1'})
plt.ylabel(r'Accuracy')
plt.xlabel(r'$SNR$')
plt.tight_layout()
plt.savefig('AMS_Robustness.pdf')
plt.show()


plt.rc('font', family='serif', serif='Computer Modern Roman', size=20)
plt.rc('text', usetex=True)
#Table Res
color=['tab:grey','orangered','steelblue']
plt.rcParams.update({'figure.autolayout': True})
ts=[1]
ms=[1,2,5]
epss=[0]
#Table Res
color=['tab:grey','orangered','steelblue']
plt.rcParams.update({'figure.autolayout': True})
lines=['-.','--','solid']
colors = pl.cm.winter(np.linspace(0,1,len(ms)))
to_plot=[]
plt.figure(figsize=(5,6))
for k in range(0,len(ms)):
    ls=lines[k]
    t=ts[0]
    c = color[k]
    to_plot = []
    std=[]
    for j in range(0,len(epss)):
        m=ms[k]
        acc = np.mean(np.load('logs/ACC_' + str(t) + '_m_' + str(m) + '_eps_'+str(epss[j])+'.npy', allow_pickle=True),axis=0)
        plt.plot(acc[0], acc[1],color=c, marker='o',linewidth=1.5, linestyle=ls,label=r'$m=$ ' + str(m))
plt.grid(True,which="both", linestyle='--')
plt.annotate(r'$\mathcal{J}^{1}$',ha = 'center', va = 'bottom',xytext = (0, 0.25),xy = (-5.5,0.35),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=.1'})
plt.annotate(r'$\mathcal{J}^{10}$',ha = 'center', va = 'bottom',xytext = (-10, 0.61),xy = (-1.4,0.6),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=-.1'})
plt.annotate(r'$\mathcal{J}^{5}$',ha = 'center', va = 'bottom',xytext = (11, 0.48),xy = (6.17,0.61),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=-.1'})
plt.ylabel(r'Accuracy')
plt.xlabel(r'$SNR$')
plt.tight_layout()
plt.savefig('AMS_Ensembling.pdf')
plt.show()
