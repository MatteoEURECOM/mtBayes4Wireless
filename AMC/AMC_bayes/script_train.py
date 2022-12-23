import main
import numpy as np

args=main.parse_args()
T=np.linspace(1,0.5,11)
M=[1,4]
for args.t in T:
    for args.m in M:
        for args.eps_tr in [0.5]:
            args.eps_te=0
            args.train_mod_type='digital'
            args.test_mod_type='digital'
            args.if_test=False
            main.main(args)
