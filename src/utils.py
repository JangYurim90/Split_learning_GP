import copy
import torch
from torchvision import datasets, transforms
from options import *
from sampling import mnist_iid, mnist_noniid , mnist_noniid_unequal

def get_dataset(args):
    """

    :param args:
    :return: train and test dataset and a user group
    """
    if args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
        ])

        train_dataset = datasets.MNIST(data_dir, train = True, download = True,
                                       transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download = True,
                                      transform=apply_transform)
        if args.iid:
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

        return train_dataset, test_dataset, user_groups

def average_weights(w):
    w_avg = copy.deepcopy(w[0]) #w[0] 깊은 복사
    for key in w_avg.keys():
        for i in range(1,len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w)) # 전체 개수로 나눠서 평균 취하기
    return w_avg

def c_aggregation(w,args):
    w_agg = copy.deepcopy(w[0])
    w_idx = copy.deepcopy(w[0])
    # (1-lamda) 에 곱하는 weight 총합
    for key in w_agg.keys():
        for i in range(1,len(w)):
            w_agg[key]+=w[i][key]
        w_agg[key] = torch.div(w_agg[key], len(w))

    for key in w_agg.keys():
        for i in range(1,len(w)):
            w_idx[i][key] = args.lamda * w_idx[i][key] + (1.-args.lamda) * w_agg[key]
    return w_agg

def h_aggregation(w,args):
    w_agg = copy.deepcopy(w[0])
    w_idx = copy.deepcopy(w[0])
    # (1-lamda) 에 곱하는 weight 총합
    for key in w_agg.keys():
        for i in range(1, len(w)):
            w_agg[key] += w[i][key]
        w_agg[key] = torch.div(w_agg[key], len(w))

    for key in w_agg.keys():
        for i in range(1, len(w)):
            w_idx[i][key] = args.lamda * w_idx[i][key] + (1. - args.lamda) * w_agg[key]
    return w_agg

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return