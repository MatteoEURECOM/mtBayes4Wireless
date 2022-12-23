import torch
from funcs.log_t import logt
from funcs.kl_div import kl
from funcs.gaussian import normal_prob,log_normal_prob
from train_and_test.testing import test 
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchbnn as bnn_lib
import tensorflow_probability as tfp

def train(args, bnn, train_loader, test_loader):
    STABLE = True
    optimizer = torch.optim.Adam(bnn.parameters(), lr=args.lr)
    actual_iter = 0
    test_nll_log=[]
    test_nllt_log = []
    test_mse_log = []
    for epoch in range(args.total_epochs):
        if(epoch%10==0):
            test_nll,test_nllt,test_mse = test(args, bnn, test_loader)
            print('epoch', epoch, ' --- t= ' + str(args.t) + ' m= ' + str(args.m))
            print('Test NLL: ' + "%.4f" % test_nll + ' --- Test NLLT: ' + "%.4f" % test_nllt+ ' --- Test MSE: ' + "%.4f" % test_mse)
            test_nll_log.append(test_nll)
            test_nllt_log.append(test_nllt)
            test_mse_log.append(test_mse)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            probs = bnn(data, args.m)
            log_p_x = torch.stack([log_normal_prob(target, p, args.sigma) for p in probs])
            p_x = torch.exp(log_p_x)
            if(args.t<=1):
                if(STABLE):
                    if (args.m==1):
                        log_avg_prob = logt(args.t, torch.mean(torch.exp(log_p_x), 0) + 10e-300)
                    else:
                        a = torch.max(log_p_x, 0)[0].detach()
                        a_exp = torch.max(p_x, 0)[0].detach() + 10e-300
                        log_avg_prob = (1 + (1 - args.t) * logt(args.t, a_exp)) * logt(args.t, torch.mean(torch.exp(log_p_x - a), 0)) + logt(args.t, a_exp)
                else:
                    log_avg_prob=logt(args.t,torch.mean(torch.exp(log_p_x), 0)+ 10e-300)
            else:
                log_avg_prob = torch.logsumexp(torch.add(log_p_x,-np.log(args.m)),axis=0)
            data_fitting_loss_term = -torch.mean(log_avg_prob)
            KL_term=bnn_lib.functional.bayesian_kl_loss(bnn)
            training_loss = data_fitting_loss_term + args.beta*KL_term
            training_loss.backward()
            optimizer.step()
            actual_iter += 1
    return test_nll_log,test_nllt_log,test_mse_log
        



