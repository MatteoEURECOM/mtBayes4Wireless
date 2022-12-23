import torch
import argparse
from data_loader.loc_data import localization_dataset
from nets.cnn import NN
from train_and_test.training import train
from train_and_test.testing import mse_avg
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def parse_args():
    parser = argparse.ArgumentParser(description='uai')
    parser.add_argument('--eps', type=float, default=0.1, help='Contamination Ratio')
    parser.add_argument('--num_neurons_hidden', type=int, default=50, help='number of neurons for BNN')
    parser.add_argument('--sigma', type=int, default=0.01   , help='Likelihood Variance')
    parser.add_argument('--tr_batch_size', type=int, default=128, help='minibatchsize for training')
    parser.add_argument('--te_batch_size', type=int, default=128, help='minibatchsize for testing')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--total_epochs', type=int, default=1000, help='total epochs for training')
    parser.add_argument('--m', type=int, default=10, help='number of multisample during training')
    parser.add_argument('--m_te', type=int, default=100, help='number of multisample for test')
    parser.add_argument('--t', type=float, default=1, help='t value for log-t')  # 0.5
    parser.add_argument('--beta', type=float, default=0, help='beta for KL term')  # 100000000
    parser.add_argument('--sigma_prior', type=float, default=1, help='prior for sigma')
    parser.add_argument('--no-cuda', action='store_true', default=True, help='disables CUDA training')
    parser.add_argument('--if_test', action='store_true', default=False, help='only testing')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    return args



def main(args):
    print('Called with args:')
    print(args)
    METER = True
    torch.manual_seed(0)
    np.random.seed(0)
    train_dataset = localization_dataset(csv_path=args.dataset + '.csv', train=True, epsilon=args.eps)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.tr_batch_size, shuffle=True, num_workers=0)
    test_dataset = localization_dataset(csv_path=args.dataset + '.csv', train=False, epsilon=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.te_batch_size, shuffle=True, num_workers=0)
    MC_reps = 1
    LOG_NLL = []
    LOG_MSE = []
    LOG_NLLT = []
    TEST_MSE = []
    for rep in range(0, MC_reps):
        bnn = NN(num_neurons_hidden=args.num_neurons_hidden, in_shape=train_dataset[:][0].shape[1]).to(args.device)
        args.m_te = 1
        test_nll, test_nllt, test_mse = train(args, bnn, train_loader, test_loader)
        LOG_NLL.append(test_nll)
        LOG_NLLT.append(test_nllt)
        LOG_MSE.append(test_mse)
        torch.save(bnn, 'saved_models/REP_' + str(rep) + '_t_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps) + '_dataset_' + str(args.dataset)+'_freq')
        bnn = torch.load('saved_models/REP_' + str(rep) + '_t_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps) + '_dataset_' + str(args.dataset)+'_freq', map_location=torch.device('cpu'))
        TEST_MSE.append(mse_avg(args, bnn, test_loader, METER))
    if (METER):
        np.save('logs/MSE_AVG_Temp_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps) + '_dataset_' + str(args.dataset) + '_METER_freq.npy', TEST_MSE)
    else:
        np.save('logs/MSE_AVG_Temp_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps) + '_dataset_' + str(args.dataset) + '_freq.npy', TEST_MSE)
    np.save('logs/LOG_Temp_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps) + '_dataset_' + str(args.dataset) + '_freq.npy', [LOG_NLL, LOG_NLLT, LOG_MSE])

if __name__ == '__main__':
    args = parse_args()
    main(args)
