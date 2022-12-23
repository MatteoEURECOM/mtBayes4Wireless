## Code of Robust Bayesian Learning for Reliable Wireless AI: Framework and Applications (https://arxiv.org/abs/2207.00300)


The repository contains 4 folder, each associated to the experiments presented in the paper:
1. Toy_Example, minimization of the different free energy criteria given a multimodal target distribution, a misspecified model (Figure 1 in the paper)  and outliers (Figure 3 in the paper).
2. AMC, Automated Modulation Classification with interference (Figure 4&5 in the paper)
3. Localization, RSSI-Based localization with malicious/imprecise reporting (Figure 7 in the paper)
4. VAE_Channel_Sim, Channel modeling with variational autoencoder (Figure 9 in the paper)

# Requirements:
The code has been written using the following packages
- Python 3.8
- numpy
- torch 1.10.1
- torchbnn 1.1
- pyproj 3.4.1    
- geopy 2.2.0
- tensorflow-probability 0.16.0

## Toy Example

The toy_example folder contains the ```toy_example.py``` that allows to load the dataset and minimize different energy criteria. 
The different energy criteria can be obtained modifying the lines:
```
''' Parameters about the free-energy'''
m=1 # multi sample parameter
t = 1  # log-t parameter
beta = 0.1  # beta parameter
```
The scripts ```plot_1.py``` and ```plot_2.py``` can be used to plot Figure 1 and 2 of the paper.

## Automated Modulation Classification

The AMC folder contains the code to run the Automated Modulation Classification task both for frequentist (AMC_freq folder) and bayesian models (AMC_bayes folder).
The dataset is too large to be uploaded on GitHub and can be downloaded from: https://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2?__hstc=24938661.5249e39e933212082be294d01b6d7bb2.1671787951336.1671787951336.1671787951336.1&__hssc=24938661.1.1671787951336&__hsfp=3042926992
Once it has been downloaded it should be placed inside the data_loader folder.
The script ```script_train.py``` (inside the AMC_freq and AMC_bayes folders) trains of all the necessary models, the results can be plot using the ```plot_ECE_ACC.py``` script.

## Localization

The Localization folder contains the code to run the RSSI-Based Localization problem for frequentist (loc_freq folder) and bayesian models (loc_bayes folder).
Inside the data_loader folder there is the compressed data that has to be unzipped before training.
The script ```script_train.py``` (inside the loc_freq and loc_bayes folders) trains of all the necessary models, the results can be plot using the ```plot_MSE.py``` script.

## VAE Channel Estimation

The VAE_Channel_Sim folder contains the code to run the VAE Channel Simulation problem. The data is included in two .mat files. The script ```script_train.py``` trains both frequentist and Bayesian models.
The results can be plotted using the ```plot.py``` script.

## Contacts 

For question feel free to contact Zecchin Matteo (zecchin@eurecom.fr)
