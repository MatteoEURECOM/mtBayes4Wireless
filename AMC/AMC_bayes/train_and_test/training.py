import torch
from funcs.log_t import logt
from funcs.kl_div import kl
from funcs.gaussian import normal_prob,log_normal_prob
from train_and_test.testing import test 
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchbnn as bnn_lib
import tensorflow_probability as tfp
# one epoch
def train(args, bnn, train_loader, test_loader,rep):
    optimizer = torch.optim.Adam(bnn.parameters(), lr=args.lr)
    actual_iter = 0
    test_nll_log=[]
    test_nllt_log = []
    test_mse_log = []
    best_nnlt=100
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)
    for epoch in range(args.total_epochs):
        if(epoch%5==0 and epoch>0):
            test_nll, test_nllt, test_mse = test(args, bnn, test_loader)
            test_nll_log.append(test_nll)
            test_nllt_log.append(test_nllt)
            test_mse_log.append(test_mse)
            print('EPOCH: ', epoch, ' --- t= ' + str(args.t) + ' m= ' + str(args.m), ' Learning Rate: ', np.asarray(scheduler.get_last_lr()))
            print('Test NLL : ' + "%.4f" % test_nll + ' --- Test NLLT: ' + "%.4f" % test_nllt + ' --- Test Acc: ' + "%.4f" % test_mse)
            train_nll, train_nllt, train_mse = test(args, bnn, train_loader)
            print('Train NLL : ' + "%.4f" % train_nll + ' --- Train NLLT: ' + "%.4f" % train_nllt + ' --- Train Acc: ' + "%.4f" % train_mse)
            if (best_nnlt > train_nllt):
                torch.save(bnn, 'saved_models/t_' + str(args.t) + '_m_' + str(args.m) + '_eps_' + str(args.eps_tr) + '_REP_' + str(rep))
            else:
                best_nnlt = train_nllt
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            avg_prob = bnn(data, args.m)
            # m: number of multisamples
            if(args.t<=1):
                data_loss = logt(args.t, torch.sum(avg_prob * target, 1))
            else:
                data_loss = torch.log(torch.sum(avg_prob * target, 1))
            data_fitting_loss_term = -torch.mean(data_loss)
            KL_term=bnn_lib.functional.bayesian_kl_loss(bnn)
            training_loss = data_fitting_loss_term +args.beta*KL_term
            training_loss.backward()
            optimizer.step()
            actual_iter += 1
        scheduler.step()
    return test_nll_log,test_nllt_log,test_mse_log



