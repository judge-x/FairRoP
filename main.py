'''
Client Selection for Federated Learning
'''
import os
import sys
import time

AVAILABLE_WANDB = False
try:
    import wandb
except ModuleNotFoundError:
    AVAILABLE_WANDB = False

import torch
import random

from data import *
from model import *
from FL_core import *
from FL_core.server import Server
from FL_core.client_selection import *
from FL_core.federated_algorithm import *
from utils import utils
from utils.argparse import get_args



def load_data(args):
    if args.dataset == 'Reddit':
        return RedditDataset(args.data_dir, args)
    elif args.dataset == 'Fed_EMNIST':
        return FederatedEMNISTDataset(args.data_dir, args)
    elif args.dataset == 'Fed_EMNIST_IID':
        return FederatedEMNISTDatasetIID(args.data_dir, args)
    elif args.dataset == 'Fed_EMNIST_NIID':
        return FederatedEMNISTDataset_nonIID(args.data_dir, args)
    elif args.dataset == 'Fed_EMNIST_POSI':
        return FederatedEMNISTDataset_Posi(args.data_dir, args)
    elif args.dataset == 'Fed_EMNIST_NIID_POSI':
        return FederatedEMNISTDataset_NIID_Posi(args.data_dir, args)
    elif args.dataset == 'Fed_CIFAR100':
        return FederatedCIFAR100Dataset(args.data_dir, args)
    elif args.dataset == 'CelebA':
        return CelebADataset(args.data_dir, args)
    elif args.dataset == 'Fed_CIFAR10_NIID':
        return PartitionedCIFAR10Dataset(args.data_dir, args)
    elif args.dataset == 'Fed_FMNIST_NIID':
        return FMNIST_NIID(args.data_dir, args)
    elif args.dataset == 'Fed_FMNIST_NIID_POSI':
        return Fed_FMNIST_NIID_POSI(args.data_dir, args)
    elif args.dataset == 'Fed_CIFAR10_POSI':
        return Fed_CIFAR10_POSI(args.data_dir, args)

def create_model(args):
    if args.dataset == 'Reddit' and args.model == 'BLSTM':
        model = BLSTM(vocab_size=args.maxlen, num_classes=args.num_classes)
    elif args.dataset == 'Fed_EMNIST_IID' and args.model == 'CNN':
        model = CNN_EMNIST()
    elif args.dataset == 'Fed_EMNIST_NIID' and args.model == 'CNN':
        model = CNN_EMNIST()
    elif args.dataset == 'Fed_EMNIST_POSI' and args.model == 'CNN':
        model = CNN_EMNIST()
    elif args.dataset == 'Fed_EMNIST_NIID_POSI' and args.model == 'CNN':
        model = CNN_EMNIST()
    elif args.dataset == 'Fed_EMNIST' in args.dataset and args.model == 'CNN':
        model = CNN_EMNIST()
    elif args.dataset == 'Fed_CIFAR100' and args.model == 'ResNet':
        model = resnet18(num_classes=args.num_classes, group_norm=args.num_gn)
        # ResNet18+GN
    elif args.dataset == 'CelebA' and args.model == 'CNN':
        model = ModelCNNCeleba()
    elif args.dataset == 'Fed_CIFAR10_NIID' or args.dataset == 'Fed_CIFAR10_POSI':
        model = CNN_CIFAR()
    elif args.dataset == 'Fed_FMNIST_NIID' or args.dataset == 'Fed_FMNIST_NIID_POSI':
        model = CNN_FMNIST()



    model = model.to(args.device)
    if args.parallel:
        model = torch.nn.DataParallel(model, output_device=0)
    return model

def create_ac(num,gramma,lr):

    return Agent(n_per_clients=num,gramma=gramma,lr=lr)



def federated_algorithm(dataset, model, args):
    train_sizes = dataset['train']['data_sizes']
    if args.fed_algo == 'FedAdam':
        return FedAdam(train_sizes, model, args=args)
    elif args.fed_algo == 'q-ffl':
        return qFFL(train_sizes, model, args=args)
    elif args.fed_algo == 'FedKrum':
        return FedKrum(train_sizes, model, args=args)
    elif args.fed_algo == 'FedPEFL':
        return FedPEFL(train_sizes, model, args=args)
    elif args.fed_algo == 'FedBulyan':
        return FedBulyan(train_sizes, model, args=args)
    elif args.fed_algo == 'Ditto':
        return Ditto(train_sizes, model)
    else:
        return FedAvg(train_sizes, model)


def client_selection_method(args):
    #total = args.total_num_client if args.num_available is None else args.num_available
    kwargs = {'total': args.total_num_client, 'device': args.device}
    if args.method == 'Random':
        return RandomSelection(**kwargs)
    elif args.method == 'AFL':
        return ActiveFederatedLearning(**kwargs, args=args)
    elif args.method == 'Cluster1':
        return ClusteredSampling1(**kwargs, n_cluster=args.num_clients_per_round)
    elif args.method == 'Cluster2':
        return ClusteredSampling2(**kwargs, dist=args.distance_type)
    elif args.method == 'Pow-d':
        assert args.num_candidates is not None
        return PowerOfChoice(**kwargs, d=args.num_candidates)
    elif args.method == 'DivFL':
        assert args.subset_ratio is not None
        return DivFL(**kwargs, subset_ratio=args.subset_ratio)
    elif args.method == 'GradNorm':
        return GradNorm(**kwargs)
    elif args.method == 'Fix':
        return FixedSelection(**kwargs)
    elif args.method == 'FairRoP':
        return FairRoP(**kwargs, eplison_greedy=args.ep_greedy,num_clients_per_round=args.num_clients_per_round)
    else:
        raise('CHECK THE NAME OF YOUR SELECTION METHOD')



if __name__ == '__main__':
    # set up
    args = get_args()
    if args.comment != '': args.comment = '-'+args.comment
    #if args.labeled_ratio < 1: args.comment = f'-L{args.labeled_ratio}{args.comment}'
    if args.fed_algo != 'FedAvg': args.comment = f'-{args.fed_algo}{args.comment}'

    # if train A2C
    ac_train=args.ac_t
    # save to wandb
    args.wandb = AVAILABLE_WANDB
    if args.wandb:
        wandb.init(
            project=f'AFL-{args.dataset}-{args.num_clients_per_round}-{args.num_available}-{args.total_num_clients}',
            name=f"{args.method}{args.comment}",
            config=args,
            dir='.',
            save_code=True
        )
        wandb.run.log_code(".", include_fn=lambda x: 'src/' in x or 'main.py' in x)

    # fix seed
    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # device setting
    if args.gpu_id == 'cpu' or not torch.cuda.is_available():
        args.device = 'cpu'
    else:
        if ',' in args.gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
        args.device = torch.device(f"cuda:{args.gpu_id[0]}")
        torch.cuda.set_device(args.device)
        print('Current cuda device is : ', torch.cuda.current_device())

    # set data
    data = load_data(args)
    args.num_classes = data.num_classes
    args.total_num_client, args.test_num_clients = data.train_num_clients, data.test_num_clients
    dataset = data.dataset

    # set model
    model = create_model(args)
    client_selection = client_selection_method(args)
    fed_algo = federated_algorithm(dataset, model, args)


    model_dir='./checkpoints'
    if ac_train==1:
        agents=create_ac(args.num_clients_per_round*2,args.ac_gramma,args.ac_lr)
    else:
        agents = create_ac(args.num_clients_per_round * 2, args.ac_gramma, args.ac_lr)
        agents.model.load_state_dict(torch.load(os.path.join(model_dir,'a2c_emnist1.pth')))
        print('load model successfully.')



    # save results
    # files = utils.save_files(args)

    ## train
    # set federated optim algorithm
    if args.poison_or_not==1:
        if args.posion_name=='label-flipping':
            print('Conduct Label-flipping attack. The poisoned index is :',data.poison_list)
            ServerExecute = Server(dataset, model, args, client_selection, fed_algo, agents,poison_index=data.poison_list)
        else:
            poison_list = random.sample([i for i in range(args.total_num_client)], int(args.total_num_client * args.poison_rate))
            print('Conduct Noise attack. The poisoned index is :',poison_list)
            ServerExecute = Server(dataset, model, args, client_selection, fed_algo, agents,poison_index=poison_list)
    else:
        ServerExecute = Server(dataset, model, args, client_selection, fed_algo, agents,poison_index=[])
    FairRoP_model=ServerExecute.train()

    #save model
    # torch.save(FairRoP_model,os.path.join(model_dir,'a2c_emnist.pth'))

