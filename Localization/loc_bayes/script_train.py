import main

args=main.parse_args()
datasets = ['sigfox_dataset_rural', 'UTS', 'UJI']
for args.dataset in datasets:
    for args.t in [1,0.96]:
        for args.m in [10]:
            for cont_ratio in [0,0.1,0.2,0.3,0.4,0.5]:
                args.eps=cont_ratio
                args.if_test=False
                main.main(args)
