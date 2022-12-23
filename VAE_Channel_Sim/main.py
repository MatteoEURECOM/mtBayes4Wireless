import torch
import argparse
from data_loader.channel_data import channel_dataset
from nets.VAE import VAE
from train_and_test.training import train
from train_and_test.testing import test,test_latent_space,compute_mmd,auroc
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from scipy.ndimage import gaussian_filter

def parse_args():
    parser = argparse.ArgumentParser(description='uai')
    parser.add_argument('--eps_tr', type=float, default=0, help='Contamination Ratio Training')
    parser.add_argument('--eps_te', type=float, default=0, help='Contamination Ratio Testing')
    parser.add_argument('--sigma', type=int, default=0.1, help='Likelihood Variance')
    parser.add_argument('--dim', type=int, default=5, help='latent space dimension')
    parser.add_argument('--size_tr', type=int, default=10e2, help='size for training')
    parser.add_argument('--size_val', type=int, default=10e2, help='size for validation')
    parser.add_argument('--size_te', type=int, default=5*10e2, help='size for testing')
    parser.add_argument('--tr_batch_size', type=int, default=64, help='minibatchsize for training')
    parser.add_argument('--te_batch_size', type=int, default=100, help='minibatchsize for testing')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--total_epochs', type=int, default=2000, help='total epochs for training')
    parser.add_argument('--m', type=int, default=2, help='number of multisample during training')
    parser.add_argument('--m_te', type=int, default=50, help='number of multisample for test')
    parser.add_argument('--t', type=float, default=1, help='t value for log-t')  # 0.5
    parser.add_argument('--beta', type=float, default=0.01, help='beta for KL term')  # 100000000
    parser.add_argument('--sigma_prior', type=float, default=1, help='prior for sigma')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--if_test', action='store_true', default=False, help='only testing')
    parser.add_argument('--num_bin', type=int, default=11, help='total number of bins for ECE')
    parser.add_argument('--BayesianDec', default=False, help='If True BNN for Dec')
    parser.add_argument('--BayesEnc', default=False, help='If True BNN for Enc')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    return args

def main(args):
    print('Called with args:')
    print(args)
    MC_reps = 3
    torch.manual_seed(0)
    LOG_NLL,LOG_ACC,LOG_NLLT,TEST,MMD,AUROC = [], [], [], [], [],[]
    if(not args.BayesianDec):
        args.m_te=1
    for rep in range(0, MC_reps):
        train_dataset = channel_dataset(mode='train', epsilon_tr=args.eps_tr, epsilon_te=args.eps_te, size_tr=args.size_tr,size_val=args.size_val,size_te=args.size_te, seed=rep)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.tr_batch_size, shuffle=True, num_workers=0)
        val_dataset = channel_dataset(mode='val', epsilon_tr=args.eps_tr, epsilon_te=args.eps_te,size_tr=args.size_tr,size_val=args.size_val,size_te=args.size_te, seed=rep)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.te_batch_size, shuffle=True, num_workers=0)
        bnn = VAE(train_dataset.X.shape[1],args.BayesEnc,args.BayesianDec,args.dim).to(args.device)
        test_nll, test_nllt, test_acc = train(args, bnn, train_loader, val_loader, rep)
        LOG_NLL.append(test_nll)
        LOG_NLLT.append(test_nllt)
        LOG_ACC.append(test_acc)
        if (args.BayesianDec):
            torch.save(bnn, 'saved_models/REP_' + str(rep) + '_temp_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps_tr) + '_dim_' + str(args.dim))
            bnn = torch.load('saved_models/REP_' + str(rep) + '_temp_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps_tr) + '_dim_' + str(args.dim), map_location=torch.device('cpu'))
        else:
            torch.save(bnn, 'saved_models/REP_' + str(rep) + '_temp_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps_tr) + '_dim_' + str(args.dim) + '_freq')
            bnn = torch.load('saved_models/REP_' + str(rep) + '_temp_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps_tr) + '_dim_' + str(args.dim) + '_freq',map_location=torch.device('cpu'))

        test_dataset = channel_dataset(mode='test', epsilon_tr=args.eps_tr, epsilon_te=0.5, size_tr=args.size_tr,size_val=args.size_val,size_te=args.size_te, seed=rep,seed_te=rep)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.te_batch_size, shuffle=True, num_workers=0)
        AUROC.append(auroc(args, bnn, test_loader))
        test_dataset = channel_dataset(mode='test', epsilon_tr=args.eps_tr, epsilon_te=0, size_tr=args.size_tr, size_val=args.size_val, size_te=args.size_te, seed=rep, seed_te=rep)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.te_batch_size, shuffle=True, num_workers=0)
        MMD.append(compute_mmd(args,bnn,test_loader))

    if (args.BayesianDec):
        np.save('logs/LOG_' + str(args.t) + '_eps_' + str(args.eps_tr) + '_m_' + str(args.m) + '_dim_' + str(args.dim) + '_.npy', [LOG_NLL, LOG_NLLT, LOG_ACC])
        np.save('logs/AUROC_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps_tr) + '.npy', AUROC)
        np.save('logs/MMD_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps_tr) + '.npy', MMD)
    else:
        np.save('logs/LOG_' + str(args.t) + '_eps_' + str(args.eps_tr) + '_m_' + str(args.m) + '_dim_' + str(args.dim) + '_freq.npy', [LOG_NLL, LOG_NLLT, LOG_ACC])
        np.save('logs/AUROC_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps_tr) + '_freq.npy', AUROC)
        np.save('logs/MMD_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps_tr) + '_freq.npy', MMD)



if __name__ == '__main__':
    args = parse_args()
    main(args)


