import argparse

ALL_METHODS = [
    'Random', 'Cluster1', 'Cluster2', 'Pow-d', 'AFL', 'DivFL', 'GradNorm', 'Fix', 'FairRoP'
]

device = 'cpu'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='3', help='gpu cuda index')
    parser.add_argument('--dataset', type=str, default='FederatedEMNIST_nonIID', help='dataset',
                        choices=['Fed_EMNIST', 'Fed_CIFAR100', 'Fed_CIFAR10_NIID', 'Fed_CIFAR10_POSI',
                                 'Fed_EMNIST_IID', 'Fed_EMNIST_NIID', 'Fed_EMNIST_POSI', 'Fed_EMNIST_NIID_POSI','Fed_FMNIST_NIID','Fed_FMNIST_NIID_POSI'])
    parser.add_argument('--data_dir', type=str, default='./dataset', help='dataset directory')
    parser.add_argument('--model', type=str, default='CNN', help='model', choices=['BLSTM', 'CNN', 'ResNet'])
    parser.add_argument('--method', type=str, default='FairRoP', help='client selection',
                        choices=ALL_METHODS)
    parser.add_argument('--fed_algo', type=str, default='FedAvg', help='Federated algorithm for aggregation',
                        choices=['FedAvg', 'FedAdam','q-ffl','FedKrum','FedPEFL','FedBulyan','FedCluster','Ditto'])

    # optimizer
    parser.add_argument('--client_optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='client optim')
    parser.add_argument('--lr_local', type=float, default=0.001, help='learning rate for client optim')
    parser.add_argument('--lr_global', type=float, default=0.01, help='learning rate for server optim')
    parser.add_argument('--wdecay', type=float, default=0, help='weight decay for optim')
    parser.add_argument('--momentum', type=float, default=0, help='momentum for SGD')

    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon for Adam')

    parser.add_argument('--alpha1', type=float, default=0.75, help='alpha1 for AFL')
    parser.add_argument('--alpha2', type=float, default=1, help='alpha2 for AFL')
    parser.add_argument('--alpha3', type=float, default=0.1, help='alpha3 for AFL')

    # training setting
    parser.add_argument('-E', '--num_epoch', type=int, default=1, help='number of epochs')
    parser.add_argument('-B', '--batch_size', type=int, default=64, help='batch size of each client data')
    parser.add_argument('-R', '--num_round', type=int, default=2000, help='total number of rounds')
    parser.add_argument('-A', '--num_clients_per_round', type=int, default=10, help='number of participated clients')
    parser.add_argument('-K', '--total_num_clients', type=int, default=50, help='total number of clients')

    parser.add_argument('-u', '--num_updates', type=int, default=None, help='number of updates')
    parser.add_argument('-n', '--num_available', type=int, default=None,
                        help='number of available clients at each round')
    parser.add_argument('-d', '--num_candidates', type=int, default=10, help='buffer size; d of power-of-choice')

    parser.add_argument('--loss_div_sqrt', action='store_true', default=False, help='loss_div_sqrt')
    parser.add_argument('--loss_sum', action='store_true', default=False, help='sum of losses')
    parser.add_argument('--num_gn', type=int, default=0, help='number of group normalization')

    parser.add_argument('--distance_type', type=str, default='L1', help='distance type for clustered sampling 2')
    parser.add_argument('--subset_ratio', type=float, default=0.1, help='subset size for DivFL')

    parser.add_argument('-al','--dirichlet_alpha', type=float, default=0.5,
                        help='ratio of data partition from dirichlet distribution')

    parser.add_argument('--min_num_samples', type=int, default=None, help='mininum number of samples')
    parser.add_argument('--schedule', type=int, nargs='+', default=[0, 5, 10, 15, 20, 30, 40, 60, 90, 140, 210, 300],
                        help='splitting points (epoch number) for multiple episodes of training')
    parser.add_argument('--maxlen', type=int, default=400, help='maxlen for NLP dataset')

    # experiment setting
    parser.add_argument('--fix_seed', action='store_true', default=False, help='fix random seed')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--parallel', action='store_true', default=False, help='use multi GPU')
    parser.add_argument('--use_mp', action='store_true', default=False, help='use multiprocessing')
    parser.add_argument('--nCPU', type=int, default=None, help='number of CPU cores for multiprocessing')
    parser.add_argument('--save_probs', action='store_true', default=False, help='save probs')
    parser.add_argument('--no_save_results', action='store_true', default=False, help='save results')
    parser.add_argument('--test_freq', type=int, default=1, help='test all frequency')

    parser.add_argument('--comment', type=str, default='', help='comment')

    parser.add_argument('--ep_greedy','-eg',type=float,default=0.8,help='MAB eplsion greedy')

    # for gaussion noise
    parser.add_argument('--sigma', '-sig', type=float, default=10, help='the privacy budget')

    # for PPO
    parser.add_argument('-alr', '--ac_lr', type=float, default=0.001, help='actor-critic learning rate')
    parser.add_argument('-g', '--ac_gramma', type=float, default=0.8, help='actor-critic reward discount gramma')
    parser.add_argument('-ac_train', '--ac_t', type=int, default=1, help='actor-critic train or test')

    # for posion
    parser.add_argument('-p', '--poison_or_not', type=int, default=0, help='whether posion the dataset')
    parser.add_argument('-p_n', '--posion_name', type=str, default="label-flipping",help='which attack will be launch.', choices=['label-flipping','noise-attack'])
    parser.add_argument('-p_rt', '--poison_rate', type=float, default=0, help='Proportion of clients poisoned')
    parser.add_argument('-dete', '--detect_or_not', type=int, default=0, help='whether posion will be detect')

    # for q-ffl
    parser.add_argument('-q', '--qffl', type=float, default=1, help='q for q-ffl')
    args = parser.parse_args()
    return args