import main_freq
import numpy as np

args=main_freq.parse_args()
T=np.linspace(1,0.5,11)
print(T)
for args.t in T:
        for args.eps_tr in [0.5]:  # fraction of corrupted samples
            args.eps_te = 0
            args.train_mod_type = 'digital'
            args.test_mod_type = 'digital'
            args.if_test = False
            main_freq.main_freq(args)

