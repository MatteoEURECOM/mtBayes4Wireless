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

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def test(args, bnn, test_loader):
    nll = 0
    corr = 0
    nllt = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(args.device), target.to(args.device)
            avg_prob = bnn(data)
            log_t_data_loss = logt(args.t, torch.sum(avg_prob * target, 1))
            log_data_loss = torch.log(torch.sum(avg_prob * target, 1))
            nll += -torch.sum(log_data_loss)
            nllt += -torch.sum(log_t_data_loss)
            pred = avg_prob.argmax(dim=1, keepdim=True)
            target_ind= target.argmax(dim=1, keepdim=True)
            # print('pred', pred)
            corr += pred.eq(target_ind.view_as(pred)).sum().item()
    test_acc = corr / len(test_loader.dataset)
    test_nll = nll/len(test_loader.dataset)
    test_nllt = nllt / len(test_loader.dataset)
    return test_nll.cpu().data.numpy(),test_nllt.cpu().data.numpy(),test_acc

def test_snrs(args, bnn, test_loader):
    acc = []
    classes=test_loader.dataset.mods
    for snr in test_loader.dataset.snrs:
        # extract classes @ SNR
        test_X_i = test_loader.dataset.X[np.where(np.array(test_loader.dataset.test_SNRs)==snr)]
        test_Y_i = test_loader.dataset.Y[np.where(np.array(test_loader.dataset.test_SNRs)==snr)]
        # estimate classes
        test_Y_i_hat = bnn(torch.from_numpy(test_X_i))
        conf = np.zeros([len(classes),len(classes)])
        confnorm = np.zeros([len(classes),len(classes)])
        for i in range(0,test_X_i.shape[0]):
            j = list(test_Y_i[i,:]).index(1)
            k = int(test_Y_i_hat[i,:].argmax(dim=0, keepdim=True))
            conf[j,k] = conf[j,k] + 1
        for i in range(0,len(classes)):
            confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        acc.append(1.0*cor/(cor+ncor))
    print("Overall Accuracy: ", cor / (cor + ncor))
    return [test_loader.dataset.snrs,acc,cor/(cor + ncor)]


def test_with_ece_tfp_snrs(args, bnn, test_loader):
    ECE=[]
    for snr in test_loader.dataset.snrs:
        # extract classes @ SNR
        test_X_i = test_loader.dataset.X[np.where(np.array(test_loader.dataset.test_SNRs) == snr)]
        test_Y_i = test_loader.dataset.Y[np.where(np.array(test_loader.dataset.test_SNRs) == snr)]
        test_Y_i_hat = bnn(torch.from_numpy(test_X_i)).detach().numpy()
        true_pred = np.argmax(test_Y_i,axis=1)
        ECE.append(tfp.stats.expected_calibration_error(args.num_bin, logits=test_Y_i_hat, labels_true=true_pred).numpy())
    print(ECE)
    return [test_loader.dataset.snrs, ECE]
def test_with_ece_tfp(args, bnn, test_loader):
    logits=[]
    true_pred=[]
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(args.device), target.to(args.device)
            logits.append(bnn(data, args.m_te))  # m: number of multisamples
            true_pred.append(target)
    logits=torch.cat(logits,axis=0)
    true_pred = torch.argmax(torch.cat(true_pred,axis=0))
    ECE=tfp.stats.expected_calibration_error(args.num_bin,logits=logits,labels_true=true_pred).numpy()

    return ECE
def test_with_ece(args, bnn, test_loader):
    bin_dict = None
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(args.device), target.to(args.device)
            avg_prob = bnn(data) # m: number of multisamples
            #pred = avg_prob.argmax(dim=1, keepdim=True)
            pred_conf, pred_class = torch.max(avg_prob, dim=1, keepdim=True)
            bin_dict = binning(bin_dict, args.num_bin, pred_conf, pred_class, target.unsqueeze(1))

    #accuracy = correct_cnt/len(test_loader.dataset)
    #print('Test Accuracy'+str(accuracy))
    ece = ece_compute(bin_dict)
    print(ece)
    return ece,bin_dict


def binning(bin_dict, num_bin, conf_mb, pred_class_mb, target_mb):
    # here m is different than number of multi-samples!
    assert conf_mb.shape[0] == pred_class_mb.shape[0] == target_mb.shape[0]
    if bin_dict is None:
        bin_dict = {}
        for m in range(num_bin):
            m += 1
            bin_dict[m] = {}
            bin_dict[m]['conf'] = 0
            bin_dict[m]['acc'] = 0
            bin_dict[m]['num'] = 0
        #print('bin dict', bin_dict)
    else:
        pass

    for ind_mb in range(target_mb.shape[0]):
        curr_conf = conf_mb[ind_mb]
        curr_pred_class = np.int(pred_class_mb[ind_mb])
        curr_target = np.int(np.argmax(target_mb[ind_mb]))
        if curr_pred_class == curr_target:
            correct = 1
        else:
            correct = 0
        for m in range(num_bin):
            m += 1
            #print('m', m)
            if (m-1)/num_bin < curr_conf <= m/num_bin:
                #print('curr conf', curr_conf)
                bin_dict[m]['conf'] += curr_conf
                bin_dict[m]['acc'] += correct
                bin_dict[m]['num'] += 1
                break
            else:
                pass
    return bin_dict

def ece_compute(bin_dict):
    total_samples_te = 0
    for m in bin_dict.keys():
        total_samples_te += bin_dict[m]['num']
    ece = 0
    for m in bin_dict.keys():
        #print('m', m)
        if bin_dict[m]['num'] == 0:
            pass
        else:
            avg_acc = bin_dict[m]['acc']/bin_dict[m]['num']
            avg_conf = bin_dict[m]['conf']/bin_dict[m]['num']
            ece += (bin_dict[m]['num']/total_samples_te)*torch.abs(avg_acc - avg_conf)
    return ece



