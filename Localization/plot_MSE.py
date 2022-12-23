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
ms=[10]
epss=[0,0.1,0.2]
datasets = ['sigfox_dataset_rural', 'UTS','UJI']
dataset = datasets[0]
lines=['-.',':','solid']
colors = pl.cm.winter(np.linspace(0,1,len(epss)))
to_plot=[]
plt.figure(figsize=(5,6))
ls = lines[0]
c = color[0]
to_plot = []
std=[]
for eps in [0,0.1,0.2,0.3,0.4,0.5]:
    m = 1
    t=1
    acc = np.load('loc_freq/logs/MSE_AVG_Temp_' + str(t) + '_m_' + str(m) + '_eps_' + str(eps) + '_dataset_' + str(dataset) + '_METER_freq.npy', allow_pickle=True)
    to_plot.append(acc[0][0])
plt.plot([0,0.1,0.2,0.3,0.4,0.5], to_plot, color=c, linewidth=2, linestyle=ls, marker='o', label=r'$m=$' + str(m) + ' $t=$' + str(t))
for k in range(0,1):
    ls=lines[k+1]
    t=ts[k]
    to_plot = []
    std=[]
    for j in range(0,len(epss)):
        c = color[k+1]
        m=ms[0]
        acc = np.load('loc_bayes/logs/MSE_AVG_Temp_' + str(t) + '_m_' + str(m) + '_eps_'+str(epss[j])+'_dataset_'+str(dataset)+'_METER.npy', allow_pickle=True)
        to_plot.append(acc[0][0])
    plt.plot(epss, to_plot,color=c, linewidth=2, linestyle=ls,marker='o',label=r'$m=$' + str(m)+' $t=$'+ str(t))
plt.grid(True,which="both", linestyle='--')
plt.yscale('log',base=10)
plt.xlabel(r'$\epsilon$')
plt.xticks([0,0.1,0.2,0.3,0.4,0.5])
plt.tight_layout()
plt.savefig('MSE_AVG_'+str(ts)+'_EPS_'+str(epss)+'_DATASET_'+dataset+'_METER.png',dpi=300)
plt.show()



dataset = datasets[1]
ls = lines[0]
c = color[0]
to_plot = []
std=[]
for eps in [0,0.1,0.2,0.3,0.4,0.5]:
    m = 1
    t=1
    acc = np.load('loc_freq/logs/MSE_AVG_Temp_' + str(t) + '_m_' + str(m) + '_eps_' + str(eps) + '_dataset_' + str(dataset) + '_METER_freq.npy', allow_pickle=True)
    to_plot.append(acc[0][0])
plt.plot([0,0.1,0.2,0.3,0.4,0.5], to_plot, color=c, linewidth=2, linestyle=ls, marker='o', label=r'$m=$' + str(m) + ' $t=$' + str(t))
for k in range(0,1):
    ls=lines[k+1]
    t=ts[k]
    to_plot = []
    std=[]
    for j in range(0,len(epss)):
        c = color[k+1]
        m=ms[0]
        acc = np.load('loc_bayes/logs/MSE_AVG_Temp_' + str(t) + '_m_' + str(m) + '_eps_'+str(epss[j])+'_dataset_'+str(dataset)+'_METER.npy', allow_pickle=True)
        to_plot.append(acc[0][0])
    plt.plot(epss, to_plot,color=c, linewidth=2, linestyle=ls,marker='o',label=r'$m=$' + str(m)+' $t=$'+ str(t))
plt.grid(True,which="both", linestyle='--')
plt.yscale('log',base=10)
plt.xlabel(r'$\epsilon$')
plt.xticks([0,0.1,0.2,0.3,0.4,0.5])
plt.tight_layout()
plt.savefig('MSE_AVG_'+str(ts)+'_EPS_'+str(epss)+'_DATASET_'+dataset+'_METER.png',dpi=300)
plt.show()


dataset = datasets[2]
ls = lines[0]
c = color[0]
to_plot = []
std=[]
for eps in [0,0.1,0.2,0.3,0.4,0.5]:
    m = 1
    t=1
    acc = np.load('loc_freq/logs/MSE_AVG_Temp_' + str(t) + '_m_' + str(m) + '_eps_' + str(eps) + '_dataset_' + str(dataset) + '_METER_freq.npy', allow_pickle=True)
    to_plot.append(acc[0][0])
plt.plot([0,0.1,0.2,0.3,0.4,0.5], to_plot, color=c, linewidth=2, linestyle=ls, marker='o', label=r'$m=$' + str(m) + ' $t=$' + str(t))
for k in range(0,1):
    ls=lines[k+1]
    t=ts[k]
    to_plot = []
    std=[]
    for j in range(0,len(epss)):
        c = color[k+1]
        m=ms[0]
        acc = np.load('loc_bayes/logs/MSE_AVG_Temp_' + str(t) + '_m_' + str(m) + '_eps_'+str(epss[j])+'_dataset_'+str(dataset)+'_METER.npy', allow_pickle=True)
        to_plot.append(acc[0][0])
    plt.plot(epss, to_plot,color=c, linewidth=2, linestyle=ls,marker='o',label=r'$m=$' + str(m)+' $t=$'+ str(t))
plt.grid(True,which="both", linestyle='--')
plt.yscale('log',base=10)
plt.xlabel(r'$\epsilon$')
plt.xticks([0,0.1,0.2,0.3,0.4,0.5])
plt.tight_layout()
plt.savefig('MSE_AVG_'+str(ts)+'_EPS_'+str(epss)+'_DATASET_'+dataset+'_METER.png',dpi=300)
plt.show()