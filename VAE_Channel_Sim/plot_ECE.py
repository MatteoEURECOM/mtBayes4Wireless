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
plt.rcParams.update({"text.usetex": True,"font.family": "sans-serif","font.sans-serif": ["Helvetica"],'font.size': 14,'lines.linewidth':2})




plt.rc('font', family='serif', serif='Computer Modern Roman', size=20)
plt.rc('text', usetex=True)
ts=[1]
ms=[1]
epss=[0]
#Table Res
plt.rcParams.update({'figure.autolayout': True})
lines=['-.','solid','..']
colors = pl.cm.winter(np.linspace(0,1,len(epss)))
to_plot=[]
for k in range(0,len(ts)):
    plt.figure(figsize=(4, 6))
    t=ts[k]
    to_plot = []
    std=[]
    m = ms[0]
    for j in range(0,len(epss)):
        acc = np.load('logs/ECE_VAL' + str(t) + '_m_' + str(m) + '_eps_'+str(epss[j])+'_freq.npy', allow_pickle=True)
        conf_num = np.mean([[a[1][i]['conf'] for i in range(1, 12)]for a in acc],axis=0)
        tot=np.sum(conf_num).numpy()
        plt.bar(np.linspace(0,1-0.5/12,len(conf_num)),conf_num/tot,1./12,color='steelblue')
        plt.xlim([0,1])
        plt.ylim([0, 0.9])
        plt.yticks(np.linspace(0, 0.8, 9))
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.grid()
        plt.savefig('ConfidecenceIDFreq.pdf')
        plt.show()

for k in range(0,len(ts)):
    plt.figure(figsize=(4, 6))
    t=ts[k]
    to_plot = []
    std=[]
    m = ms[0]
    for j in range(0,len(epss)):
        acc = np.load('logs/ECE_VAL' + str(t) + '_m_' + str(m) + '_eps_'+str(epss[j])+'_freq_OOD.npy', allow_pickle=True)
        conf_num = np.mean([[a[1][i]['conf'] for i in range(1, 12)]for a in acc],axis=0)
        tot=np.sum(conf_num).numpy()
        plt.bar(np.linspace(0.5/12,1-0.5/12,len(conf_num)),conf_num/tot,1./12,color='tab:grey')
        plt.xlim([0,1])
        plt.ylim([0, 0.9])
        plt.yticks(np.linspace(0, 0.8, 9))
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.grid()
        plt.savefig('ConfidecenceOODFreq.pdf')
        plt.show()
plt.show()


plt.rc('font', family='serif', serif='Computer Modern Roman', size=18)
plt.rc('text', usetex=True)
#Table Res
color=['tab:grey','orangered','steelblue']
plt.rcParams.update({'figure.autolayout': True})
ts=[1]
ms=[1]
epss=[0.5]
ECE=0
for k in range(0,len(ts)):
    t = ts[k]
    to_plot = []
    std = []
    m = ms[0]
    for j in range(0,len(epss)):
        acc = np.load('logs/ECE_VAL' + str(t) + '_m_' + str(m) + '_eps_'+str(epss[j])+'.npy', allow_pickle=True)
        conf=[np.asarray([np.float(a[1][i]['conf']) for i in range(1, 12)]) for a in acc]
        accuracy=[np.asarray([a[1][i]['acc'] for i in range(1, 12)])for a in acc]
        tot = [np.asarray([a[1][i]['num'] for i in range(1, 12)]) for a in acc]
        for i in range(0,3):
            ECE+=np.sum(np.abs(accuracy[i]-conf[i])/np.sum(tot[i]))
            print(ECE)
            plt.title('tmp: ' + str(t))
            plt.plot(np.abs(accuracy[i])/tot[i])
            plt.plot(np.abs(conf[i]) / tot[i])
            plt.show()
print(ECE/3)