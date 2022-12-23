import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import torch
import scipy.interpolate as interp

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def movingaverage(interval, window_size):
    interval=np.concatenate((interval,interval[-window_size+1:]))
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'valid')

iter=300
plt.rcParams.update({"text.usetex": True,"font.family": "serif","font.serif": ["Computer Modern Roman"],'font.size': 20,'lines.linewidth':2})

#Table Res
color=['tab:grey','orangered','steelblue']
plt.rc('text', usetex=True)
datasets = ['sigfox_dataset_rural', 'sigfox_dataset_antwerp', 'lorawan_antwerp_2019_dataset','UTS']
dataset = datasets[0]
lines=['solid','-.',':']


ts=[1,0.6,0.4]
ms=[5]
epss=[0.25]
rep=0
colors = pl.cm.winter(np.linspace(0,1,len(epss)))
to_plot=[]

legends=[r'$\mathcal{J}^1_4$',r'$\mathcal{J}^{0.6}_4$',r'$\mathcal{J}^{0.4}_4$']

ts=[1,0.9,0.7]
ms=[4]
epss=[0.2]
for k in range(0,len(ts)):
    ls=lines[k]
    m=ms[0]
    t=ts[0]
    to_plot = []
    for j in range(0,len(epss)):
        c = color[k]
        acc = np.load('logs/AUROC_' + str(ts[k]) + '_m_' + str(ms[0]) + '_eps_' + str(epss[j]) + '.npy', allow_pickle=True)

        lens=[len(acc[i][1]) for i in range(0,3)]
        tper=[acc[i][0].squeeze() for i in range(0,3)]
        fper = [acc[i][1].squeeze() for i in range(0, 3)]
        '''
        for i in range(0,3):
            plt.plot(tper[i], fper[i], color=color[k], label=r'$\mathcal{J}^1_4$')
        plt.show()'''
        max_length = 5000
        print(max_length)
        inter=[interp.interp1d(acc[i][0].squeeze(), acc[i][1].squeeze()) for i in range(0,3)]
        mean = np.mean([inter[i](np.linspace(0,1,max_length)) for i in range(0,3)],axis=0)
        std = np.std([inter[i](np.linspace(0,1,max_length)) for i in range(0,3)],axis=0)
        plt.plot(np.linspace(0,1,max_length+1), np.hstack([0,mean]), color=color[k], label=legends[k])
        plt.fill_between(np.linspace(0,1,max_length), (mean- 2*std/ 3.), (mean+ 2*std/ 3.), color=color[k], alpha=.3)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.grid()
plt.xlabel(r'False Positive Rate')
plt.ylabel(r'True Positive Rate')
plt.title(r'Receiver Operating Characteristic Curve')
plt.legend()
plt.savefig('ROC_contaminated.pdf')
plt.show()

color=['tab:grey','orangered','steelblue']

fig = plt.figure(figsize=(8,5))
gs = fig.add_gridspec(2, hspace=0.1)
axs = gs.subplots(sharex=True)
plt.subplots_adjust(bottom=0.15)
plt.tight_layout()
to_plot_means = []
to_plot_stds = []
REP=3
ts=[1,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]
for t in ts:
    acc = np.load('logs/AUROC_' + str(t) + '_m_4_eps_0.2.npy', allow_pickle=True)
    aurocs = [acc[i][2] for i in range(0, 3)]
    to_plot_means.append(np.mean(aurocs))
    to_plot_stds.append(2 * np.std(aurocs) / 3.)
axs[1].plot(ts, to_plot_means,c='tab:blue', marker='o',linestyle='-',markersize=5)
axs[1].fill_between(ts,(to_plot_means + to_plot_stds/np.sqrt(REP)),(to_plot_means - to_plot_stds/np.sqrt(REP)), color='tab:blue', alpha=.3)
to_plot_means = []
to_plot_stds = []
for t in ts:
    acc = np.load('logs/AUROC_1_m_1_eps_0.2_freq.npy', allow_pickle=True)
    aurocs = [acc[i][2] for i in range(0, 3)]
    to_plot_means.append(np.mean(aurocs))
    to_plot_stds.append(2 * np.std(aurocs) / 3.)
axs[1].plot(ts, to_plot_means,c='black', marker='o',linestyle=':',markersize=5)
axs[1].fill_between(ts,(to_plot_means + to_plot_stds/np.sqrt(REP)),(to_plot_means - to_plot_stds/np.sqrt(REP)), color='black', alpha=.3)
axs[1].annotate(r'Frequentist',ha = 'center', va = 'bottom',xytext = (0.6, 0.92 ),xy = (0.75,0.86),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=-.1'})
axs[1].annotate(r'$(4,t)$-Robust Bayesian',ha = 'center', va = 'bottom',xytext = (0.87, 0.975),xy = (0.85,0.895),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=-.1'})
axs[1].set_ylabel(r'AUROC')
axs[1].set_xlim([0.5,1])
axs[1].grid()


to_plot_means = []
to_plot_stds = []
for t in ts:
    acc = np.load('logs/MMD_' + str(t) + '_m_4_eps_0.2.npy', allow_pickle=True)
    to_plot_means.append(np.mean(acc))
    to_plot_stds.append(2 * np.std(acc) / 3.)
axs[0].plot(ts, to_plot_means, c='tab:blue', marker='o', linestyle='--', markersize=5)
axs[0].fill_between(ts, (to_plot_means + to_plot_stds / np.sqrt(REP)), (to_plot_means - to_plot_stds / np.sqrt(REP)), color='tab:blue', alpha=.3)
to_plot_means = []
to_plot_stds = []
for t in ts:
    acc = np.load('logs/MMD_1_m_1_eps_0.2_freq.npy', allow_pickle=True)
    to_plot_means.append(np.mean(acc))
    to_plot_stds.append(2 * np.std(acc) / 3.)
axs[0].plot(ts, to_plot_means,c='black', marker='o',linestyle=':',markersize=5)
axs[0].fill_between(ts,(to_plot_means + to_plot_stds/np.sqrt(REP)),(to_plot_means - to_plot_stds/np.sqrt(REP)), color='black', alpha=.3)
axs[0].annotate(r'Frequentist',ha = 'center', va = 'bottom',xytext = (0.6, 0.016 ),xy = (0.75,0.019),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=.1'})
axs[0].annotate(r'$(4,t)$-Robust Bayesian',ha = 'center', va = 'bottom',xytext = (0.87, 0.011 ),xy = (0.85,0.0175),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=-.1'})
axs[1].set_xlabel(r'$t$')
axs[0].set_ylabel(r'MMD')
axs[0].grid()

plt.savefig('MMD+AUROC.pdf')
plt.show()





ts=[1]
ms=[1,2,3]
epss=[0]
plt.figure(figsize=(5, 5))
for k in range(0,len(ms)):
    ls=lines[k]
    m=ms[0]
    t=ts[0]
    to_plot = []
    for j in range(0,len(epss)):
        c = color[k]
        acc = np.load('logs/AUROC_' + str(ts[0]) + '_m_' + str(ms[k]) + '_eps_' + str(epss[j]) +'.npy', allow_pickle=True)
        lens=[len(acc[i][1]) for i in range(0,3)]
        tper=[acc[i][0].squeeze() for i in range(0,3)]
        fper = [acc[i][1].squeeze() for i in range(0, 3)]

        for i in range(0,3):
            plt.plot(tper[i], fper[i], color=color[k], label=r'$\mathcal{J}^1_4$')
        plt.show()
        max_length = 5000
        print(max_length)
        inter=[interp.interp1d(acc[i][0].squeeze(), acc[i][1].squeeze()) for i in range(0,3)]
        mean = np.mean([inter[i](np.linspace(0,1,max_length)) for i in range(0,3)],axis=0)
        std = np.std([inter[i](np.linspace(0,1,max_length)) for i in range(0,3)],axis=0)
        plt.plot(np.linspace(0,1,max_length+1), np.hstack([0,mean]), color=color[k], label=legends[k])
        plt.fill_between(np.linspace(0,1,max_length), (mean- 2*std/ 3.), (mean+ 2*std/ 3.), color=color[k], alpha=.3)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.grid()
plt.xlabel(r'False Positive Rate')
plt.ylabel(r'True Positive Rate')
plt.title(r'Receiver Operating Characteristic Curve')
plt.legend()
plt.savefig('ROC_contaminated.pdf')
plt.show()

plt.grid()
plt.show()


for k in range(0,len(ts)):
    ls=lines[k]
    m=ms[0]
    t=ts[0]
    to_plot = []
    for j in range(0,len(epss)):
        c = color[k]
        acc = np.load('logs/AUROC_' + str(ts[k]) + '_m_' + str(ms[0]) + '_eps_' + str(epss[j]) + '.npy', allow_pickle=True)

        lens=[len(acc[i][1]) for i in range(0,3)]
        tper=[acc[i][0].squeeze() for i in range(0,3)]
        fper = [acc[i][1].squeeze() for i in range(0, 3)]
        '''
        for i in range(0,3):
            plt.plot(tper[i], fper[i], color=color[k], label=r'$\mathcal{J}^1_4$')
        plt.show()'''
        max_length = 5000
        print(max_length)
        inter=[interp.interp1d(acc[i][0].squeeze(), acc[i][1].squeeze()) for i in range(0,3)]
        mean = np.mean([inter[i](np.linspace(0,1,max_length)) for i in range(0,3)],axis=0)
        std = np.std([inter[i](np.linspace(0,1,max_length)) for i in range(0,3)],axis=0)
        plt.plot(np.linspace(0,1,max_length+1), np.hstack([0,mean]), color=color[k], label=legends[k])
        plt.fill_between(np.linspace(0,1,max_length), (mean- 2*std/ 3.), (mean+ 2*std/ 3.), color=color[k], alpha=.3)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.grid()
plt.xlabel(r'False Positive Rate')
plt.ylabel(r'True Positive Rate')
plt.title(r'Receiver Operating Characteristic Curve')
plt.legend()
plt.savefig('ROC_contaminated.pdf')
plt.show()


ts=[1]
ms=[1,2,3]
epss=[0]
plt.figure(figsize=(5, 5))
for k in range(0,len(ms)):
    ls=lines[k]
    m=ms[0]
    t=ts[0]
    to_plot = []
    for j in range(0,len(epss)):
        c = color[k]
        acc = np.load('logs/AUROC_' + str(ts[0]) + '_m_' + str(ms[k]) + '_eps_' + str(epss[j]) +'.npy', allow_pickle=True)
        lens=[len(acc[i][1]) for i in range(0,3)]
        tper=[acc[i][0].squeeze() for i in range(0,3)]
        fper = [acc[i][1].squeeze() for i in range(0, 3)]
        '''
        for i in range(0,3):
            plt.plot(tper[i], fper[i], color=color[k], label=r'$\mathcal{J}^1_4$')
        plt.show()'''
        max_length = 5000
        print(max_length)
        inter=[interp.interp1d(acc[i][0].squeeze(), acc[i][1].squeeze()) for i in range(0,3)]
        mean = np.mean([inter[i](np.linspace(0,1,max_length)) for i in range(0,3)],axis=0)
        std = np.std([inter[i](np.linspace(0,1,max_length)) for i in range(0,3)],axis=0)
        plt.plot(np.linspace(0,1,max_length+1), np.hstack([0,mean]), color=color[k], label=legends[k])
        plt.fill_between(np.linspace(0,1,max_length), (mean- 2*std/ 3.), (mean+ 2*std/ 3.), color=color[k], alpha=.3)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.grid()
plt.xlabel(r'False Positive Rate')
plt.ylabel(r'True Positive Rate')
plt.title(r'Receiver Operating Characteristic Curve')
plt.legend()
plt.savefig('ROC_contaminated.pdf')
plt.show()

ts=[1]
ms=[1,2,3,4,5,6]
epss=[0]
rep=0
colors = pl.cm.winter(np.linspace(0,1,len(epss)))
to_plot=[]
plt.figure(figsize=(6,4))
color=['tab:grey','orangered','steelblue']
th=3
to_plot_means = []
to_plot_stds = []
for k in range(0,len(ms)):
    m=ms[k]
    t=ts[0]
    for j in range(0,len(epss)):
        acc = np.load('logs/MMD_' + str(t) + '_m_' + str(m) + '_eps_' + str(epss[j]) + '.npy', allow_pickle=True)
        to_plot_means.append(np.mean(acc))
        to_plot_stds.append(2*np.std(acc)/3.)
plt.errorbar(ms, to_plot_means, to_plot_stds, marker='o',capsize=1)
plt.xlabel(r'$m$')
plt.ylabel(r'MMD')
plt.grid()
plt.show()
