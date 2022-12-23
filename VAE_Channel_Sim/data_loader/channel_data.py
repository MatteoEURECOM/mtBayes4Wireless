import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
import sionna as si
import tensorflow as tf
import scipy.io


def get_samples_100(num,index=0):
    # UT and BS panel arrays
    X = np.transpose(scipy.io.loadmat('TDL-A-100.mat')['CIR'])
    X =  np.float32(X[index:index+num, 0:128])
    return X
def get_samples_300(num,index=0):
    X=[]
    if(num>0):
        X = np.vstack([np.transpose(scipy.io.loadmat('TDL-A-300.mat')['CIR'])[:,0:128]])#,np.transpose(scipy.io.loadmat('TDL-A-1000.mat')['CIR'])[:,0:128]])
        np.random.shuffle(X)
        X =  np.float32(X[index:index+num, 0:128])
    return X

class channel_dataset(torch.utils.data.Dataset):
    def __init__(self,mode,epsilon_tr,epsilon_te,size_tr,size_val,size_te,seed,seed_te=1):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        if(epsilon_tr>0):
            self.X = np.vstack([get_samples_100(int(size_tr*(1-epsilon_tr))),get_samples_300(int(size_tr*epsilon_tr))])
        else:
            self.X = get_samples_100(int(size_tr))
        mean = np.mean(self.X)
        std = np.std(self.X)
        self.X = (self.X ) / std
        if mode=='val':
            np.random.seed(seed)
            tf.random.set_seed(seed)
            if(epsilon_tr>0):
                self.X =  np.vstack([get_samples_100(int(size_val*(1-epsilon_te)),int(size_tr*(1-epsilon_tr))),get_samples_300(int(size_val*epsilon_te),int(size_tr*epsilon_tr))])
            else:
                self.X = get_samples_100(int(size_val),int(size_tr))
            self.X = (self.X ) / std
        elif mode=='test':
            np.random.seed(seed_te)
            tf.random.set_seed(seed_te)
            if(epsilon_te>0):
                self.X = np.vstack([get_samples_100(int(size_te*(1-epsilon_te)),int(size_val*(1-epsilon_te)+size_tr*(1-epsilon_tr))),get_samples_300(int(size_te*epsilon_te),int(size_val*epsilon_te+size_tr*epsilon_tr))])
                self.OOD = np.hstack([np.ones(int(size_te * (1 - epsilon_te))), np.zeros(int(size_te * epsilon_te))])
            else:
                self.X =get_samples_100(int(size_te),int(size_val+size_tr))
                self.OOD = np.hstack([np.ones(int(size_te * (1 - epsilon_te)))])
            self.X = (self.X ) / std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index,:]
