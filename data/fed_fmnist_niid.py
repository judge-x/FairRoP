import pickle
import random

import torchvision.datasets as D
import torchvision.transforms as T
from torch.utils.data import TensorDataset
import torch
import numpy as np
import os

import pandas as pd
import matplotlib.pyplot as plt


class FMNIST_NIID(object):
    def __init__(self, data_dir, args):
        '''
        partitioned CIFAR10 datatset according to a Dirichlet distribution
        '''
        self.num_classes = 10
        self.train_num_clients = 100 if args.total_num_clients is None else args.total_num_clients
        self.test_num_clients = 100 if args.total_num_clients is None else args.total_num_clients
        self.balanced = False
        self.alpha = args.dirichlet_alpha
        self.seed = 2023
        self.hist_color = '#4169E1'


        self._init_data(data_dir)
        print(f'Total number of users: {self.train_num_clients}')

    def _init_data(self, data_dir):
        file_name = os.path.join(os.path.join(data_dir, 'FashionMNIST'),
                                 'FashionMNIST_preprocessed_2.pickle')
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)
        else:
            matrix = np.random.dirichlet([self.alpha] * self.train_num_clients, size=self.num_classes)

            train_data_local_dict, train_data_local_num_dict = self.partition_FMNIST_dataset(data_dir, matrix,
                                                                                             train=True)
            test_data_local_dict, test_data_local_num_dict = self.partition_FMNIST_dataset(data_dir, matrix,
                                                                                           train=False)

            dataset = {}
            dataset['train'] = {
                'data_sizes': train_data_local_num_dict,
                'data': train_data_local_dict,
            }
            dataset['test'] = {
                'data_sizes': test_data_local_num_dict,
                'data': test_data_local_dict,
            }


            # with open(file_name,'wb') as f:
            #     pickle.dump(dataset,f)

        self.dataset = dataset

    def partition_FMNIST_dataset(self, data_dir, matrix, train):
        """Partition dataset into `n_clients`.
        Each client i has matrix[k, i] of data of class k"""

        dataset = D.FashionMNIST(os.path.join(data_dir, 'FMNIST'), train=train,
                            download=True, )

        n_clients = self.train_num_clients if train else self.test_num_clients

        list_clients_X = [[] for i in range(n_clients)]
        list_clients_y = [[] for i in range(n_clients)]
        train_labels=np.array(dataset.targets)
        client_idcs=self.dirichlet_split_noniid(train_labels,matrix)


        for idx_client in range(n_clients):
            for id_sample in client_idcs[idx_client]:
                list_clients_X[idx_client] += [dataset.data[id_sample].unsqueeze(0).numpy()]
                list_clients_y[idx_client] += [dataset.targets[id_sample]]
            list_clients_X[idx_client] = np.array(list_clients_X[idx_client])

        size = len(list_clients_X)
        # result={}
        data = {}
        data_len = {}
        for idx in range(len(list_clients_X)):
            X = torch.Tensor(list_clients_X[idx])
            Y = torch.tensor(list_clients_y[idx])
            data[idx] = TensorDataset(X, Y)
            data_len[idx] = len(list_clients_y[idx])
        # result['data']=data
        # result['data_sizes']=data_len
        return data, data_len

    def dirichlet_split_noniid(self,train_labels, matrix):
        '''
        参数为 alpha 的 Dirichlet 分布将数据索引划分为 n_clients 个子集
        '''
        # 总类别数
        n_classes = train_labels.max() + 1

        class_idcs = [np.argwhere(train_labels == y).flatten()
                      for y in range(n_classes)]

        # 定义一个空列表作最后的返回值
        client_idcs = [[] for _ in range(self.train_num_clients)]
        # 记录N个client分别对应样本集合的索引
        for c, fracs in zip(class_idcs, matrix):
            # np.split按照比例将类别为k的样本划分为了N个子集
            # for i, idcs 为遍历第i个client对应样本集合的索引
            for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                client_idcs[i] += [idcs]

        client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

        return client_idcs
