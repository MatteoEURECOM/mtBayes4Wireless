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
plt.rcParams.update({"text.usetex": True,"font.family": "sans-serif","font.sans-serif": ["Computer Modern Roman"],'font.size': 14,'lines.linewidth':2})




plt.rc('font', family='serif', serif='Computer Modern Roman', size=18)
plt.rc('text', usetex=True)

fig = plt.figure(figsize=(8,5))
gs = fig.add_gridspec(2, hspace=0.1)
axs = gs.subplots(sharex=True)

ts=np.linspace(1,0.5,11)#,0.4,0.3,0.2,0.1,0]
ms=[1]
epss=[0.5]
REP=5
acc_plt=[]
acc_std=[]
ece_plt=[]
ece_std=[]
for k in range(0,len(ts)):
    t = ts[k]
    to_plot = []
    std = []
    m = ms[0]
    for j in range(0,len(epss)):
        ECE=[]
        acc = np.load('logs/ACC_' + str(t) + '_m_' + str(m) + '_eps_' + str(epss[j]) + '.npy', allow_pickle=True)
        acc_plt.append(np.mean([a[1] for a in acc]))
        acc_std.append(np.std([a[1] for a in acc]))
        acc = np.load('logs/ECE_VAL' + str(t) + '_m_' + str(m) + '_eps_'+str(epss[j])+'.npy', allow_pickle=True)
        conf = [np.asarray([np.float(a[1][i]['conf']) for i in range(1, 12)]) for a in acc]
        conf_num = np.mean([[a[1][i]['conf'] for i in range(1, 12)] for a in acc], axis=0)
        accuracy = [np.asarray([a[1][i]['acc'] for i in range(1, 12)]) for a in acc]
        tot = [np.asarray([a[1][i]['num'] for i in range(1, 12)]) for a in acc]
        for i in range(0, REP):
            ECE.append(np.sum(np.abs(accuracy[i] - conf[i]) / np.sum(tot[i])))
        ece_plt.append(np.mean(ECE))
        ece_std.append(np.std(ECE))

ms=[4]

acc_plt_1=[]
acc_std_1=[]
ece_plt_1=[]
ece_std_1=[]
for k in range(0,len(ts)):
    t = ts[k]
    to_plot = []
    std = []
    m = ms[0]
    for j in range(0,len(epss)):
        print(t)
        ECE=[]
        acc = np.load('logs/ACC_' + str(t) + '_m_' + str(m) + '_eps_' + str(epss[j]) + '.npy', allow_pickle=True)
        acc_plt_1.append(np.mean([a[1] for a in acc]))
        acc_std_1.append(np.std([a[1] for a in acc]))
        acc = np.load('logs/ECE_VAL' + str(t) + '_m_' + str(m) + '_eps_'+str(epss[j])+'.npy', allow_pickle=True)
        conf = [np.asarray([np.float(a[1][i]['conf']) for i in range(1, 12)]) for a in acc]
        conf_num = np.mean([[a[1][i]['conf'] for i in range(1, 12)] for a in acc], axis=0)
        accuracy = [np.asarray([a[1][i]['acc'] for i in range(1, 12)]) for a in acc]
        tot = [np.asarray([a[1][i]['num'] for i in range(1, 12)]) for a in acc]
        for i in range(0,REP):
            ECE.append(np.sum(np.abs(accuracy[i] - conf[i]) / np.sum(tot[i])))
        ece_plt_1.append(np.mean(ECE))
        ece_std_1.append(np.std(ECE))
acc_plt_2=[]
ece_plt_2=[]
ece_std_2=[]
acc_std_2=[]
t=1
acc = np.load('logs/ACC_' + str(t) + '_m_1_eps_' + str(epss[j]) + '_freq.npy', allow_pickle=True)[0:5]
acc_plt_2.append(np.mean([a[1] for a in acc]))
acc_std_2.append(np.std([a[1] for a in acc]))
acc = np.load('logs/ECE_VAL' + str(t) + '_m_1_eps_'+str(epss[j])+'_freq.npy', allow_pickle=True)[0:5]
conf = [np.asarray([np.float(a[1][i]['conf']) for i in range(1, 12)]) for a in acc]
conf_num = np.mean([[a[1][i]['conf'] for i in range(1, 12)] for a in acc], axis=0)
accuracy = [np.asarray([a[1][i]['acc'] for i in range(1, 12)]) for a in acc]
tot = [np.asarray([a[1][i]['num'] for i in range(1, 12)]) for a in acc]
ECE=[]
for i in range(0, 5):
    ECE.append(np.sum(np.abs(accuracy[i] - conf[i]) / np.sum(tot[i])))
ece_plt_2.append(np.mean(ECE))
ece_std_2.append(np.std(ECE))
axs[0].plot(ts,acc_plt,c='tab:red', marker='o',linestyle='--',markersize=5)
axs[0].plot(ts,acc_plt_1,c='tab:blue',marker='o',markersize=5)
axs[0].set_ylabel(r'Accuracy')
axs[1].plot(ts,ece_plt,c='tab:red',marker='o',linestyle='--',markersize=5)
axs[1].plot(ts,ece_plt_1,c='tab:blue',marker='o',markersize=5)
axs[1].fill_between(ts, (ece_plt + ece_std/np.sqrt(REP)),(ece_plt - ece_std/np.sqrt(REP)), color='tab:red', alpha=.3)
axs[1].fill_between(ts, (ece_plt_1 + ece_std_1/np.sqrt(REP)),(ece_plt_1 - ece_std_1/np.sqrt(REP)), color='tab:blue', alpha=.3)
axs[1].fill_between(ts, (ece_plt_2 + ece_std_2/np.sqrt(REP)),(ece_plt_2 - ece_std_2/np.sqrt(REP)), color='tab:grey', alpha=.3)
axs[0].fill_between(ts, (acc_plt + acc_std/np.sqrt(REP)),(acc_plt - acc_std/np.sqrt(REP)), color='tab:red', alpha=.3)
axs[0].fill_between(ts, (acc_plt_1 + acc_std_1/np.sqrt(REP)),(acc_plt_1 - acc_std_1/np.sqrt(REP)), color='tab:blue', alpha=.3)
axs[0].fill_between(ts, (acc_plt_2 + acc_std_2/np.sqrt(REP)),(acc_plt_2 - acc_std_2/np.sqrt(REP)), color='tab:grey', alpha=.3)
axs[0].plot(ts,np.repeat(acc_plt_2,len(acc_plt_1)),c='black',marker='o',linestyle=':',markersize=5)
axs[1].plot(ts,np.repeat(ece_plt_2,len(acc_plt_1)),c='black',marker='o',linestyle=':',markersize=5)
axs[0].annotate(r'$(4,t)$-Robust Bayesian ',ha = 'center', va = 'bottom',xytext = (0.73, 0.71),xy = (0.9,0.68),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=-.1'})
axs[0].annotate(r'$(1,t)$-Robust Bayesian ',ha = 'center', va = 'bottom',xytext = (0.85, 0.51),xy = (0.9,0.58),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=-.1'})
axs[0].annotate(r'Frequentist ',ha = 'center', va = 'bottom',xytext = (0.6, 0.51),xy = (0.7,0.61),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=.1'})
axs[1].annotate(r'Frequentist ',ha = 'center', va = 'bottom',xytext = (0.585, 0.015),xy =(0.7,0.10),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=-.1'})
axs[1].annotate(r'$(4,t)$-Robust Bayesian ',ha = 'center', va = 'bottom',xytext = (0.75, 0.13),xy = (0.9,0.090),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=-.1'})
axs[1].annotate(r'$(1,t)$-Robust Bayesian ',ha = 'center', va = 'bottom',xytext = (0.8, 0.16),xy = (0.95,0.15),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=-.1'})
axs[1].set_xlabel(r'$t$')
axs[1].set_ylabel(r'ECE')
axs[1].set_xticks(ts)
axs[1].grid()
axs[1].set_xlim([0.5,1])
axs[0].set_ylim([0.5,0.75])
axs[0].grid()
plt.subplots_adjust(bottom=0.15)
plt.tight_layout()
plt.savefig('Acc+ECE_new_labels.pdf')
plt.show()


plt.figure(figsize=(7,3))

plt.plot(ts,ece_plt_1,c='tab:red',marker='o',markersize=5)
plt.fill_between(ts, (ece_plt_1 + ece_std_1/np.sqrt(2)),(ece_plt_1 - ece_std_1/np.sqrt(2)), color='tab:red', alpha=.3)
plt.plot(ts,np.repeat(ece_plt_2,11),c='black',marker='o',linestyle=':',markersize=5)
plt.fill_between(ts, (np.repeat(ece_plt_2,11) + np.repeat(ece_std_2,11)/np.sqrt(3)),(np.repeat(ece_plt_2,11) -  np.repeat(ece_std_2,11)/np.sqrt(3)), color='black', alpha=.3)
plt.xlabel(r'$t$')
plt.ylabel(r'ECE')
plt.xticks(ts)
plt.grid()
plt.xlim([0,1])
plt.tight_layout()
plt.savefig('ECE.pdf')
plt.show()




ts=[1]
ms=[1]
epss=[0.5]
plt.rc('font', family='serif', serif='Computer Modern Roman', size=18)
plt.rc('text', usetex=True)
#Table Res
color=['tab:grey','orangered','steelblue']
plt.rcParams.update({'figure.autolayout': True})
ECE=0
acc_plt=[]
ece_plt=[]
for k in range(0,len(ts)):
    t = ts[k]
    to_plot = []
    std = []
    m = ms[0]
    for j in range(0,len(epss)):
        ECE=0
        acc = np.load('logs/ECE_VAL' + str(t) + '_m_' + str(m) + '_eps_'+str(epss[j])+'.npy', allow_pickle=True)
        conf = [np.asarray([np.float(a[1][i]['conf']) for i in range(1, 12)]) for a in acc]
        conf_num = np.mean([[a[1][i]['conf'] for i in range(1, 12)] for a in acc], axis=0)
        accuracy = [np.asarray([a[1][i]['acc'] for i in range(1, 12)]) for a in acc]
        tot = [np.asarray([a[1][i]['num'] for i in range(1, 12)]) for a in acc]
        for i in range(2, 3):
            plt.figure(figsize=(6, 6))
            ECE += np.sum(np.abs(accuracy[i] - conf[i]) / np.sum(tot[i]))
            plt.grid()
            plt.plot(np.linspace(0, 1, len(conf_num)), np.linspace(0.5 / 12, 1 - 0.5 / 12, len(conf_num)), linestyle='--', color='black',label='Perfect Calibration')
            plt.bar(np.linspace(0.5 / 12, 1 - 0.5 / 12, len(conf_num)), np.abs(conf[i]) / tot[i], 1. / 12, color='red', alpha=0.65,label='Confidence')
            plt.bar(np.linspace(0.5 / 12, 1 - 0.5 / 12, len(conf_num)), np.abs(accuracy[i]) / tot[i], 1. / 12, color='blue', alpha=0.65,label='Accuracy')
            plt.legend()
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.savefig('ECE_plot_t_'+str(t)+'_m_'+str(m)+'.pdf')
            plt.title(r'$t= $' + str(t) + r', $\epsilon= $' + str(epss[j])+r', ECE= '+str( np.sum(np.abs(accuracy[i] - conf[i]) / np.sum(tot[i]))))
            plt.show()
        ece_plt.append(ECE/2)
    print(str(round(ECE/2,4)))

ts=[1]
ms=[4]
epss=[0]
for k in range(0,len(ts)):
    plt.figure(figsize=(6, 4))
    t=ts[k]
    to_plot = []
    std=[]
    m = ms[0]
    for j in range(0,len(epss)):
        acc = np.load('logs/ECE_VAL' + str(t) + '_m_' + str(m) + '_eps_'+str(epss[j])+'.npy', allow_pickle=True)
        conf_num = np.mean([[a[1][i]['conf'] for i in range(1, 12)]for a in acc],axis=0)
        tot=np.sum(conf_num).numpy()
        plt.bar(np.linspace(0,1-0.5/12,len(conf_num)),conf_num/tot,1./12,color='steelblue')
        plt.xlim([0, 1])
        plt.yticks(np.linspace(0, 0.6, 5))
        acc = np.load('logs/ECE_VAL' + str(t) + '_m_' + str(m) + '_eps_'+str(epss[j])+'_OOD.npy', allow_pickle=True)
        conf_num = np.mean([[a[1][i]['conf'] for i in range(1, 12)] for a in acc], axis=0)
        tot = np.sum(conf_num).numpy()
        plt.bar(np.linspace(0.5 / 12, 1 - 0.5 / 12, len(conf_num)), conf_num / tot, 1. / 12, color='tab:grey')
        plt.xlim([0, 1])
        plt.ylim([0, 0.9])
        plt.yticks(np.linspace(0, 0.8, 9))
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.grid()
        plt.savefig('ConfidecenceOODBayes'+str(t)+'.pdf')
        plt.show()
plt.show()


