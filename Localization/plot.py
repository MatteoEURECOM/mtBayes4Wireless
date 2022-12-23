import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def movingaverage(interval, window_size):
    interval=np.concatenate((interval,interval[-window_size+1:]))
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'valid')


iter=500
plt.rcParams.update({"text.usetex": True,"font.family": "sans-serif","font.sans-serif": ["Helvetica"],'font.size': 14,'lines.linewidth':2})
#Table Res
color=['tab:grey','orangered','steelblue']
window=10
plt.rcParams.update({'figure.autolayout': True})
n_ths=9
ths=np.linspace(0.1,0.9,n_ths)
ts=[1,0.98,0.96]
ms=[10]
epss=[0.5]
datasets = ['sigfox_dataset_rural', 'sigfox_dataset_antwerp', 'lorawan_antwerp_2019_dataset','UTS']
dataset = datasets[0]
lines=['solid','-.',':']

ts=[1,0.98,0.96]
ms=[10]
epss=[0,0.1,0.2,0.3,0.4,0.5]
rep=0
colors = pl.cm.winter(np.linspace(0,1,len(epss)))
to_plot=[]
plt.figure(figsize=(6,4))
color=['tab:grey','orangered','steelblue']
th=3
for k in range(0,len(ts)):
    ls=lines[k]
    m=ms[0]
    t=ts[k]
    to_plot = []
    for j in range(0,len(epss)):
        c = color[k]
        acc = np.load('logs/RES_Temp_' + str(t) + '_m_' + str(m) + '_eps_'+str(epss[j])+'_dataset_'+str(dataset)+'_METER.npy', allow_pickle=True)
        to_plot.append(acc[1,th])
    plt.plot(epss, to_plot,color=c, linewidth=1.5, linestyle=ls,marker='o',label=r'$m=$' + str(m)+' $t=$'+ str(t))
plt.annotate(r'$\mathcal{J}^{10}$',ha = 'center', va = 'bottom',xytext = (0.32, 0.065),xy = (0.35,0.097),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=.1'})
plt.annotate(r'$\mathcal{J}_{0.98}^{10}$',ha = 'center', va = 'bottom',xytext = (0.25, 0.05),xy = (0.26,0.024),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=-.1'})
plt.annotate(r'$\mathcal{J}_{0.96}^{10}$',ha = 'center', va = 'bottom',xytext = (0.45, 0.05),xy = (0.35, 0.023),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=.1'})
plt.grid()
plt.xlim([0,0.5])
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'Acc')
plt.tight_layout()
plt.savefig('Plots/MSE_'+str(ts)+'_EPS_'+str(epss)+'_DATASET_'+dataset+'.png',dpi=300)
plt.show()


dataset = datasets[2]
ts=[1,0.98,0.96]
ms=[10]
epss=[0,0.1,0.2,0.3,0.4,0.5]
rep=0
colors = pl.cm.winter(np.linspace(0,1,len(epss)))
to_plot=[]
plt.figure(figsize=(6,4))
color=['tab:grey','orangered','steelblue']
th=3
for k in range(0,len(ts)):
    ls=lines[k]
    m=ms[0]
    t=ts[k]
    to_plot = []
    for j in range(0,len(epss)):
        c = color[k]
        acc = np.load('logs/RES_Temp_' + str(t) + '_m_' + str(m) + '_eps_'+str(epss[j])+'_dataset_'+str(dataset)+'_METER.npy', allow_pickle=True)
        to_plot.append(acc[1,th])
    plt.plot(epss, to_plot,color=c, linewidth=1.5, linestyle=ls,marker='o',label=r'$m=$' + str(m)+' $t=$'+ str(t))
plt.annotate(r'$\mathcal{J}^{10}$',ha = 'center', va = 'bottom',xytext = (0.32, 0.065),xy = (0.35,0.097),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=.1'})
plt.annotate(r'$\mathcal{J}_{0.98}^{10}$',ha = 'center', va = 'bottom',xytext = (0.25, 0.05),xy = (0.26,0.024),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=-.1'})
plt.annotate(r'$\mathcal{J}_{0.96}^{10}$',ha = 'center', va = 'bottom',xytext = (0.45, 0.05),xy = (0.35, 0.023),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=.1'})
plt.grid()
plt.xlim([0,0.5])
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'Acc')
plt.tight_layout()
plt.savefig('Plots/MSE_'+str(ts)+'_EPS_'+str(epss)+'_DATASET_'+dataset+'.png',dpi=300)
plt.show()


ts=[1,0.98,0.96]
ms=[10]
epss=[0,0.1,0.2,0.3,0.4,0.5]
rep=0
colors = pl.cm.winter(np.linspace(0,1,len(epss)))
to_plot=[]
plt.figure(figsize=(6,4))
dataset = datasets[1]
for k in range(0,len(ts)):
    ls=lines[k]
    m=ms[0]
    t=ts[k]
    to_plot = []
    for j in range(0,len(epss)):
        c = color[k]
        acc = np.load('logs/RES_Temp_' + str(t) + '_m_' + str(m) + '_eps_'+str(epss[j])+'_dataset_'+str(dataset)+'_METER.npy', allow_pickle=True)
        to_plot.append(acc[1,th])
    plt.plot(epss, to_plot,color=c, linewidth=1.5, linestyle=ls,marker='o',label=r'$m=$' + str(m)+' $t=$'+ str(t))
plt.annotate(r'$\mathcal{J}^{10}$',ha = 'center', va = 'bottom',xytext = (0.32, 0.12),xy = (0.35,0.18),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=.1'})
plt.annotate(r'$\mathcal{J}_{0.98}^{10}$',ha = 'center', va = 'bottom',xytext = (0.42, 0.09),xy = (0.46,0.058),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=-.1'})
plt.annotate(r'$\mathcal{J}_{0.96}^{10}$',ha = 'center', va = 'bottom',xytext = (0.25, 0.09),xy = (0.22, 0.047),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=.1'})
plt.grid()
plt.xlim([0,0.5])
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'Acc')
plt.tight_layout()
plt.savefig('Plots/MSE_'+str(ts)+'_EPS_'+str(epss)+'_DATASET_'+dataset+'.png',dpi=300)
plt.show()



ts=[1,0.98,0.96]
ms=[10]
epss=[0,0.1,0.2,0.3,0.4,0.5]
rep=0
colors = pl.cm.winter(np.linspace(0,1,len(epss)))
to_plot=[]
plt.figure(figsize=(6,4))
dataset = datasets[0]
for k in range(0,len(ts)):
    ls=lines[k]
    m=ms[0]
    t=ts[k]
    to_plot = []
    for j in range(0,len(epss)):
        c = color[k]
        acc = np.load('logs/RES_Temp_' + str(t) + '_m_' + str(m) + '_eps_'+str(epss[j])+'_dataset_'+str(dataset)+'_METER.npy', allow_pickle=True)
        to_plot.append(acc[1,th])
    plt.plot(epss, to_plot,color=c, linewidth=1.5, linestyle=ls,marker='o',label=r'$m=$' + str(m)+' $t=$'+ str(t))
plt.annotate(r'$\mathcal{J}^{10}$',ha = 'center', va = 'bottom',xytext = (0.32, 0.065),xy = (0.35,0.117),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=.1'})
plt.annotate(r'$\mathcal{J}_{0.98}^{10}$',ha = 'center', va = 'bottom',xytext = (0.32, 0.05),xy = (0.35,0.015),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=-.1'})
plt.annotate(r'$\mathcal{J}_{0.96}^{10}$',ha = 'center', va = 'bottom',xytext = (0.45, 0.05),xy = (0.4, 0.006),arrowprops = {'facecolor' : 'black','width':0.3,'headwidth':4,'headlength':5,'connectionstyle':'arc3,rad=.1'})
plt.grid()
plt.xlim([0,0.5])
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'Acc')
plt.tight_layout()
plt.savefig('Plots/MSE_'+str(ts)+'_EPS_'+str(epss)+'_DATASET_'+dataset+'.png',dpi=300)
plt.show()

rep=0
colors = pl.cm.winter(np.linspace(0,1,len(ms)))
for eps in [0]:
    for k in range(0,len(ts)):
        ls=lines[k]
        t=ts[k]
        for j in range(0,len(ms)):
            c = colors[j]
            m=ms[j]
            acc = np.load('logs/RES_Temp_' + str(t) + '_m_' + str(m) + '_eps_'+str(eps)+'_dataset_'+str(dataset)+'.npy', allow_pickle=True)
            plt.plot(ths, acc[1,:],color=c , linewidth=1.5, linestyle=ls,marker='o',label=r'$m=$' + str(m)+' $t=$'+ str(t))
    plt.grid()
    plt.legend()
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'Accuracy')
    plt.tight_layout()
    plt.savefig('Plots/Accuracy_'+str(ts)+'_EPS_'+str(epss)+'.png',dpi=300)
    plt.show()
