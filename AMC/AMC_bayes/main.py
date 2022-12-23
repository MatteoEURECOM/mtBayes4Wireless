import torch
import argparse
from data_loader.ams_data import ams_dataset
from nets.bnn_multisample import BNN
from train_and_test.training import train
from train_and_test.testing import test_snrs,test_with_ece,test_with_ece_tfp,test_with_ece_tfp_snrs
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def parse_args():
    parser = argparse.ArgumentParser(description='uai')
    parser.add_argument('--eps_tr', type=float, default=0, help='Contamination Ratio Training')
    parser.add_argument('--eps_te', type=float, default=0, help='Contamination Ratio Testing')
    parser.add_argument('--tr_batch_size', type=int, default=128, help='minibatchsize for training')
    parser.add_argument('--te_batch_size', type=int, default=1024, help='minibatchsize for testing')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for training')
    parser.add_argument('--total_epochs', type=int, default=150, help='total epochs for training')
    parser.add_argument('--m', type=int, default=5, help='number of multisample during training')
    parser.add_argument('--m_te', type=int, default=5, help='number of multisample for test')
    parser.add_argument('--t', type=float, default=1, help='t value for log-t')  # 0.5
    parser.add_argument('--beta', type=float, default=.1, help='beta for KL term')
    parser.add_argument('--no-cuda', action='store_true', default=True, help='disables CUDA training')
    parser.add_argument('--if_test', action='store_true', default=True, help='only testing')
    parser.add_argument('--num_bin', type=int, default=11, help='total number of bins for ECE')
    parser.add_argument('--train_mod_type', default='digital', help='type of modulations in train data')
    parser.add_argument('--test_mod_type', default='digital', help='type of modulations in test data')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    return args

def main(args):
    print('Called with args:')
    print(args)
    MC_reps =10
    LOG_NLL,LOG_ACC,LOG_NLLT,ECE,ACC = [], [], [], [], []
    torch.manual_seed(0)
    np.random.seed(0)
    train_dataset = ams_dataset(mode='train', epsilon_tr=args.eps_tr,epsilon_te=args.eps_te,mod_type=args.train_mod_type,seed=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.tr_batch_size, shuffle=True, num_workers=0)
    test_dataset = ams_dataset(mode='test', epsilon_tr=args.eps_tr,epsilon_te=args.eps_te, mod_type=args.test_mod_type,seed=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.te_batch_size, shuffle=True, num_workers=0)
    val_dataset = ams_dataset(mode='val', epsilon_tr=args.eps_tr,epsilon_te=args.eps_te, mod_type=args.train_mod_type,seed=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.te_batch_size, shuffle=True, num_workers=0)    
    for rep in range(0, MC_reps):
        bnn = BNN().to(args.device)
        args.m_te = args.m
        # train
        args.beta=100./len(train_dataset)
        test_nll, test_nllt, test_acc = train(args, bnn, train_loader, val_loader,rep)
        LOG_NLL.append(test_nll)
        LOG_NLLT.append(test_nllt)
        LOG_ACC.append(test_acc)
        # test trained model
        bnn = torch.load('saved_models/t_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps_tr) + '_REP_' + str(rep), map_location=torch.device('cpu'))
        ECE.append(test_with_ece(args, bnn, test_loader))
        ACC.append(test_snrs(args, bnn, test_loader))
    np.save('logs/ACC_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps_tr) + '.npy', ACC)
    np.save('logs/ECE_VAL' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps_tr) + '.npy', ECE)
    np.save('logs/LOG_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps_tr) + '.npy', [LOG_NLL, LOG_NLLT, LOG_ACC])

if __name__ == '__main__':
    args = parse_args()
    main(args)
