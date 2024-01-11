'''
Reference:
    https://github.com/Accenture/Labs-Federated-Learning/tree/clustered_sampling
'''
import pickle

import torchvision.datasets as D
import torchvision.transforms as T
from torch.utils.data import TensorDataset
import torch
import numpy as np
import os
import random


import pandas as pd
import matplotlib.pyplot as plt

class Fed_CIFAR10_POSI(object):
    def __init__(self, data_dir, args):
        '''
        partitioned CIFAR10 datatset according to a Dirichlet distribution
        '''
        self.num_classes = 10
        self.poison_classes = 9

        self.train_num_clients = 50 if args.total_num_clients is None else args.total_num_clients
        self.test_num_clients = 50 if args.total_num_clients is None else args.total_num_clients
        self.balanced = True 
        self.alpha = args.dirichlet_alpha
        self.seed=2023
        self.hist_color='#4169E1'

        self.poison_rate=args.poison_rate
        self.poison_list=[]

        self._init_data(data_dir)
        print(f'Total number of users: {self.train_num_clients}')
    
    def _init_data(self, data_dir):
        file_name = os.path.join(os.path.join(data_dir,'PartitionedCIFAR10'), 'PartitionedCIFAR10_preprocessed_2.pickle')
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)
        else:
            matrix = np.random.dirichlet([self.alpha] * self.num_classes, size=self.train_num_clients)

            train_data = {}
            test_data = {}

            train_data = self.partition_CIFAR_dataset_train(data_dir, matrix, train=True)
            test_data = self.partition_CIFAR_dataset_test(data_dir, matrix, train=False)
            # test_data=train_data

            dataset = {
                'train': train_data, 
                'test' : test_data
            }

            # with open(file_name,'wb') as f:
            #     pickle.dump(dataset,f)


        self.dataset = dataset


    def partition_CIFAR_dataset_train(self, data_dir, matrix, train):
        """Partition dataset into `n_clients`.
        Each client i has matrix[k, i] of data of class k"""

        # transform = [
        #     T.ToTensor(),
        #     T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        # ]
        transform_normal = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_aug = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        dataset = D.CIFAR10(os.path.join(data_dir,'PartitionedCIFAR10'), train=train, transform=transform_aug, download=True,)

        n_clients = self.train_num_clients if train else self.test_num_clients

        # generate poisoned index
        poison_rate = self.poison_rate
        thr_index = int(n_clients * poison_rate)
        self.poison_list = random.sample([i for i in range(n_clients)], thr_index)

        list_clients_X = [[] for i in range(n_clients)]
        list_clients_y = [[] for i in range(n_clients)]

        if self.balanced:
            n_samples = [500] * n_clients
        elif not self.balanced and train:
            n_samples = [100] * int(0.1*n_clients) + [250] * int(0.3*n_clients) + [500] * int(0.3*n_clients) + [750] * int(0.2*n_clients) + [1000] * int(0.1*n_clients)
        elif not self.balanced and not train:
            n_samples = [20*2] * int(0.1*n_clients) + [50*2] * int(0.3*n_clients) + [100*2] * int(0.3*n_clients) + [150*2] * int(0.2*n_clients) + [200*2] * int(0.1*n_clients)

        list_idx = []
        for k in range(self.num_classes):

            idx_k = np.where(np.array(dataset.targets) == k)[0]
            list_idx += [idx_k]

        for idx_client, n_sample in enumerate(n_samples):

            clients_idx_i = []
            client_samples = 0

            for k in range(self.num_classes):

                if k < self.num_classes:
                    samples_digit = int(matrix[idx_client, k] * n_sample)
                if k == self.num_classes:
                    samples_digit = n_sample - client_samples
                client_samples += samples_digit

                clients_idx_i = np.concatenate(
                    (clients_idx_i, np.random.choice(list_idx[k], samples_digit))
                )

            clients_idx_i = clients_idx_i.astype(int)

            for idx_sample in clients_idx_i:

                list_clients_X[idx_client] += [dataset.data[idx_sample]]

                if idx_client in self.poison_list and dataset.targets[idx_sample] in range(self.poison_classes):
                    list_clients_y[idx_client] += [random.randint(self.poison_classes-1, self.num_classes-1)]
                else:
                    list_clients_y[idx_client] += [dataset.targets[idx_sample]]
                # list_clients_y[idx_client] += [dataset.targets[idx_sample]]

            list_clients_X[idx_client] = np.array(list_clients_X[idx_client])

    
        return {
            'data': {idx: TensorDataset(torch.Tensor(list_clients_X[idx]).permute(0, 3, 1, 2), torch.tensor(list_clients_y[idx])) for idx in range(len(list_clients_X))}, # (list_clients_X, list_clients_y),
            'data_sizes': {idx: len(list_clients_y[idx]) for idx in range(len(list_clients_X))}
        }
    
    def partition_CIFAR_dataset_test(self, data_dir, matrix, train): 
            """Partition dataset into `n_clients`.
            Each client i has matrix[k, i] of data of class k"""

            # transform = [
            #     T.ToTensor(),
            #     T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            # ]
            transform_normal = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            transform_aug = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            dataset = D.CIFAR10(os.path.join(data_dir,'PartitionedCIFAR10'), train=train, transform=transform_aug, download=True,)

            n_clients = self.train_num_clients if train else self.test_num_clients

            list_clients_X = [[] for i in range(n_clients)]
            list_clients_y = [[] for i in range(n_clients)]

            if self.balanced:
                n_samples = [500] * n_clients
            elif not self.balanced and train:
                n_samples = [100] * int(0.1*n_clients) + [250] * int(0.3*n_clients) + [500] * int(0.3*n_clients) + [750] * int(0.2*n_clients) + [1000] * int(0.1*n_clients)
            elif not self.balanced and not train:
                n_samples = [20*2] * int(0.1*n_clients) + [50*2] * int(0.3*n_clients) + [100*2] * int(0.3*n_clients) + [150*2] * int(0.2*n_clients) + [200*2] * int(0.1*n_clients)

            list_idx = []
            for k in range(self.num_classes):

                idx_k = np.where(np.array(dataset.targets) == k)[0]
                list_idx += [idx_k]

            for idx_client, n_sample in enumerate(n_samples):

                clients_idx_i = []
                client_samples = 0

                for k in range(self.num_classes):

                    if k < self.num_classes:
                        samples_digit = int(matrix[idx_client, k] * n_sample)
                    if k == self.num_classes:
                        samples_digit = n_sample - client_samples
                    client_samples += samples_digit

                    clients_idx_i = np.concatenate(
                        (clients_idx_i, np.random.choice(list_idx[k], samples_digit))
                    )

                clients_idx_i = clients_idx_i.astype(int)

                for idx_sample in clients_idx_i:
                    list_clients_X[idx_client] += [dataset.data[idx_sample]]
                    list_clients_y[idx_client] += [dataset.targets[idx_sample]]

                list_clients_X[idx_client] = np.array(list_clients_X[idx_client])

            return {
                'data': {idx: TensorDataset(torch.Tensor(list_clients_X[idx]).permute(0, 3, 1, 2), torch.tensor(list_clients_y[idx])) for idx in range(len(list_clients_X))}, # (list_clients_X, list_clients_y),
                'data_sizes': {idx: len(list_clients_y[idx]) for idx in range(len(list_clients_X))}
            }

