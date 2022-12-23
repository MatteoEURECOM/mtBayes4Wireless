import torch
from funcs.log_t import logt
from funcs.kl_div import kl
from funcs.gaussian import normal_prob,log_normal_prob
from train_and_test.testing import test 
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchbnn as bnn_lib


def train(args, bnn, train_loader, test_loader,rep):
    optimizer = torch.optim.Adam(bnn.parameters(), lr=args.lr)
    actual_iter = 0
    test_nll_log=[]
    test_nllt_log = []
    test_mse_log = []
    best_nnlt = 100
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    for epoch in range(args.total_epochs):
        if(epoch%50==0 and epoch>0):
            test_nll,test_nllt,test_mse = test(args, bnn, test_loader)
            print('EPOCH: ', epoch, ' --- t= ' + str(args.t) + ' m= ' + str(args.m))
            print('Test NLL : ' + "%.4f" % test_nll + ' --- Test NLLT: ' + "%.4f" % test_nllt+ ' --- Test Acc: ' + "%.4f" % test_mse)
            test_nll_log.append(test_nll)
            test_nllt_log.append(test_nllt)
            test_mse_log.append(test_mse)
        for batch_idx, data in enumerate(train_loader):
            data= data.to(args.device)
            optimizer.zero_grad()
            probs,mus,log_vars = bnn(data,args.m)
            log_p_x = torch.stack([log_normal_prob(data, p, args.sigma) for p in probs])
            # m: number of multisamples
            if (args.t < 1):
                    log_avg_prob = logt(args.t,torch.mean(torch.exp(log_p_x),axis=0))
            else:
                log_avg_prob = torch.logsumexp(torch.add(log_p_x, -np.log(args.m)), axis=0)
            data_fitting_loss_term = -torch.mean(log_avg_prob)
            KL_term_hid = -0.5 * torch.mean(1 + log_vars[0] - mus[0].pow(2) - log_vars[0].exp())
            KL_term = bnn_lib.functional.bayesian_kl_loss(bnn)
            training_loss = data_fitting_loss_term + args.beta*KL_term + 0.1*KL_term_hid
            training_loss.backward()
            optimizer.step()
            actual_iter += 1
        scheduler.step()
    return test_nll_log,test_nllt_log,test_mse_log
        



