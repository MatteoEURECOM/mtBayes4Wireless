import main
import numpy as np

args=main.parse_args()
args.BayesianDec = True
args.BayesianEnc = False
args.if_test = True
T=np.linspace(1,0.5,11)
for args.t in T:
     for args.m in [4]:
         args.eps_te= 0.2
         args.eps_tr = 0.2
         main.main(args)

args.BayesianDec = False
args.BayesianEnc = False
args.if_test = True
T=np.linspace(1,0.5,11)
for args.t in T:
     for args.m in [4]:
         args.eps_te= 0.2
         args.eps_tr = 0.2
         main.main(args)
