import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class localization_dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, train,epsilon):
        self.dataset_name=csv_path
        scaler = MinMaxScaler()
        if(csv_path=='sigfox_dataset_rural.csv'):
            self.df = pd.read_csv('data_loader/sigfox_dataset_rural.csv')
            self.RSSI =  self.df.iloc[:, 1:137].to_numpy()
            self.loc =  self.df.iloc[:, -2:].to_numpy()
        elif (csv_path == 'UTS.csv'):
            self.RSSI = np.concatenate((pd.read_csv('data_loader/UTS_training.csv').to_numpy()[:,0:588],pd.read_csv('data_loader/UTS_test.csv').to_numpy()[:,0:588]))
            self.loc = np.concatenate((pd.read_csv('data_loader/UTS_training.csv').to_numpy()[:,589:591],pd.read_csv('data_loader/UTS_test.csv').to_numpy()[:,589:591]))
        elif (csv_path == 'UJI.csv'):
            self.RSSI = np.concatenate((pd.read_csv('data_loader/UJITrain.csv').to_numpy()[:, 0:520], pd.read_csv('data_loader/UJIVal.csv').to_numpy()[:, 0:520]))
            self.loc = np.concatenate((pd.read_csv('data_loader/UJITrain.csv').to_numpy()[:, 520:522], pd.read_csv('data_loader/UJIVal.csv').to_numpy()[:, 520:522]))
        train_ratio=0.8
        n_train=int(len(self.loc)*train_ratio)
        self.cov_scale=scaler.fit(self.RSSI)
        self.RSSI=self.cov_scale.transform(self.RSSI)
        self.label=scaler.fit(self.loc)
        self.loc = self.label.transform(self.loc)
        n_corr_samples=int(len(self.loc) * epsilon)
        np.random.seed(0)
        self.train_samples=np.random.choice(len(self.loc),n_train,replace=False)
        self.test_samples=np.delete(np.arange(0,len(self.loc)),self.train_samples)
        self.corrupted_ind = np.random.choice(n_train,n_corr_samples,replace=False)
        if(train):
            self.RSSI=np.float32(self.RSSI[ self.train_samples])
            self.loc=np.float32(self.loc[ self.train_samples])
            self.loc[self.corrupted_ind]=np.random.random((n_corr_samples,2))
        else:
            self.RSSI = np.float32(self.RSSI[self.test_samples])
            self.loc = np.float32(self.loc[self.test_samples])
        if(csv_path == 'UTS.csv' or csv_path == 'UJI.csv'):
            self.RSSI=1-self.RSSI
    def __len__(self):
        return len(self.loc)

    def __getitem__(self, index):
        return self.RSSI[index,:], self.loc[index,:]
