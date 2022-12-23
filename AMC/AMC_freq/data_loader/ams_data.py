import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer

class ams_dataset(torch.utils.data.Dataset):
    def __init__(self,mode,epsilon_tr,epsilon_te,mod_type,seed):
        np.random.seed(seed)
        Xd = pd.read_pickle('data_loader/RML2016.10a_dict.pkl')
        self.snrs, self.mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
        X,lbl = [],[]
        '''Organizing Data by SNR levels and modulation schemes'''
        self.snrs = [s for s in self.snrs if s>0]
        self.analog_mods = ['AM-DSB', 'AM-SSB', 'WBFM']
        self.digital_mods = [m for m in self.mods if m not in self.analog_mods]
        for mod in  self.digital_mods:  #Use only digital modulations during training
            for snr in  self.snrs:
                X.append(Xd[(mod, snr)])
                for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
        X = np.vstack(X)
        n_examples = X.shape[0]
        n_train = n_examples * 0.3   #30% for training
        n_val = n_examples * 0.2    #20% for validation
        n_test=n_examples-n_train-n_val     #50% for testing
        self.train_idx = np.random.choice(range(0, int(n_examples)), size=int(n_train), replace=False)
        self.val_idx = np.random.choice(list(set(range(0, n_examples)) - set(self.train_idx)),size=int(n_val),replace=False)
        self.test_idx = list(set(range(0, n_examples)) - set(self.train_idx)-set(self.val_idx))
        self.X = X[self.train_idx]   #Training data set
        '''One hot encoding'''
        lb = LabelBinarizer()
        lb.fit([l[0] for l in lbl])
        one_hot = lb.transform([l[0] for l in lbl])
        self.Y = one_hot[self.train_idx]  #One-hot encoded labels
        if(epsilon_tr>0):
            '''Interference in case of epsilon_tr>0'''
            corr_ind = np.random.choice(range(0, int(n_train)), size=int(n_train * epsilon_tr), replace=False)
            int_ind = np.random.choice(range(0, self.X.shape[0]), size=int(n_train * epsilon_tr), replace=True)
            self.X[corr_ind]= 0.2*self.X[corr_ind]+1.*self.X[int_ind] #Signal superposition
        ''' Data set normalization '''
        mean = np.mean(self.X)
        std = np.std(self.X)
        self.X = (self.X - mean) / std #Normalization
        if mode=='val':
            self.X = X[self.val_idx]
            self.Y = one_hot[self.val_idx]
            epsilon_tr=0
            if (epsilon_tr > 0):
                '''Interference in case of epsilon_tr>0. Validation data is also corrupted by interference. '''
                corr_ind = np.random.choice(range(0, int(n_val)), size=int(n_val*epsilon_tr), replace=False)
                int_ind = np.random.choice(range(0, int(n_val)), size=int(n_val * epsilon_tr), replace=False)
                self.X[corr_ind]= 0.2*self.X[corr_ind]+1*self.X[int_ind]
            self.X = (self.X - mean) / std #Normalization
        elif mode=='test':
            self.test_SNRs = np.asarray([ lbl[i][1] for i in self.test_idx])
            self.X = X[ self.test_idx]
            self.Y = one_hot[ self.test_idx]
            if (epsilon_te > 0):
                '''Just in case one wants to test the performance for corrupted test sets'''
                corr_ind = np.random.choice(range(0, int(n_test)), size=int(n_test * epsilon_te), replace=False)
                int_ind = np.random.choice(range(0, int(n_test)), size=int(n_test * epsilon_te), replace=False)
                self.X[corr_ind] = self.X[corr_ind] + self.X[int_ind]
            self.X = (self.X - mean) / std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index,:,:], self.Y[index,:]
