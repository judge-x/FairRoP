import os
import random
import h5py
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset


import torchvision.datasets as D
import torchvision.transforms as T


class FederatedEMNISTDataset_NIID_Posi:
    def __init__(self, data_dir, args):
        '''
        partitioned EMNIST datatset according to a Dirichlet distribution
        known class: digits (10)
        unknown class: characters (52) -> label noise
        '''
        self.num_classes = 62
        self.poison_class = 52
        self.train_num_clients = 50 if args.total_num_clients is None else args.total_num_clients
        self.test_num_clients = 50 if args.total_num_clients is None else args.total_num_clients
        self.balanced = True
        self.alpha = args.dirichlet_alpha
        self.seed=2023
        self.hist_color='#4169E1'
        self.poison_or_not = args.poison_or_not
        self.poison_rate = args.poison_rate
        self.poison_list = []
        self._init_data(data_dir)
        print(f'Total number of users: {self.train_num_clients}')



    def _init_data(self, data_dir):
        data_dir=os.path.join(data_dir, 'FederatedEMNIST_nonIID')
        file_name = os.path.join(data_dir, 'FederatedEMNIST_preprocessed_nonIID.pickle')
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)
        else:
            matrix = np.random.dirichlet([self.alpha] * self.num_classes, size=self.train_num_clients)


            train_data_local_dict, train_data_local_num_dict = self.partition_EMNIST_dataset_train(data_dir, matrix, train=True)
            test_data_local_dict, test_data_local_num_dict = self.partition_EMNIST_dataset_test(data_dir, matrix, train=False)

            dataset = {}
            dataset['train'] = {
                'data_sizes': train_data_local_num_dict,
                'data': train_data_local_dict,
            }
            dataset['test'] = {
                'data_sizes': test_data_local_num_dict,
                'data': test_data_local_dict,
            }

            # with open(file_name, 'wb') as f:
            #     pickle.dump(dataset, f)

        self.dataset=dataset

    def partition_EMNIST_dataset_train(self, data_dir, matrix, train):
        """Partition dataset into `n_clients`.
        Each client i has matrix[k, i] of data of class k"""

        transform_aug = T.Compose([T.ToTensor(),T.Normalize((0.1307,),(0.3081,))])

        dataset = D.EMNIST(data_dir, train=train, transform=transform_aug,
                            download=True, split="byclass",)

        n_clients = self.train_num_clients if train else self.test_num_clients


        # init poisoned clients index
        poison_rate = self.poison_rate
        thr_index = int(n_clients * poison_rate)
        self.poison_list = random.sample([i for i in range(n_clients)], thr_index)
        # posion_class=[(self.num_classes-i) for i in range(self.poisoned_class)]

            # print("poisoned client are",self.poison_list)

        list_clients_X = [[] for i in range(n_clients)]
        list_clients_y = [[] for i in range(n_clients)]

        if self.balanced:
            n_samples = [500] * n_clients
        elif not self.balanced and train:
            n_samples = [100] * int(0.1 * n_clients) + [250] * int(0.3 * n_clients) + [500] * int(0.3 * n_clients) + [
                750] * int(0.2 * n_clients) + [1000] * int(0.1 * n_clients)
            
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
                list_clients_X[idx_client] += [dataset.data[idx_sample].unsqueeze(0).numpy()]

                if idx_client in self.poison_list:
                    tmp=dataset.targets[idx_sample]
                    if dataset.targets[idx_sample].item() in range(self.poison_class):
                        tmp=torch.tensor(random.randint(self.poison_class-1, self.num_classes-1))
                        list_clients_y[idx_client] += [tmp]
                    else:
                        list_clients_y[idx_client] += [dataset.targets[idx_sample]]
                else:
                    list_clients_y[idx_client] += [dataset.targets[idx_sample]]

            list_clients_X[idx_client] = np.array(list_clients_X[idx_client])
        # tmp= torch.tensor([item.detach().numpy() for item in list_clients_X[0]])

        size=len(list_clients_X)
        # result={}
        data={}
        data_len={}
        for idx in range(len(list_clients_X)):
            X=torch.Tensor(list_clients_X[idx])
            Y=torch.tensor(list_clients_y[idx])
            data[idx]=TensorDataset(X, Y)
            data_len[idx]=len(list_clients_y[idx])
        # result['data']=data
        # result['data_sizes']=data_len
        return data,data_len

    def partition_EMNIST_dataset_test(self, data_dir, matrix, train):
        """Partition dataset into `n_clients`.
        Each client i has matrix[k, i] of data of class k"""

        transform_aug = T.Compose([T.ToTensor(),T.Normalize((0.1307,),(0.3081,))])

        dataset = D.EMNIST(data_dir, train=train, transform=transform_aug,
                            download=True, split="byclass",)

        n_clients = self.train_num_clients if train else self.test_num_clients

        list_clients_X = [[] for i in range(n_clients)]
        list_clients_y = [[] for i in range(n_clients)]

        if self.balanced:
            n_samples = [500] * n_clients
        elif not self.balanced and not train:
            n_samples = [40] * int(0.1 * n_clients) + [100] * int(0.3 * n_clients) + [200] * int(0.3 * n_clients) + [
                300] * int(0.2 * n_clients) + [400] * int(0.1 * n_clients)

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
                list_clients_X[idx_client] += [dataset.data[idx_sample].unsqueeze(0).numpy()]
                list_clients_y[idx_client] += [dataset.targets[idx_sample]]

            list_clients_X[idx_client] = np.array(list_clients_X[idx_client])

        # result={}
        data={}
        data_len={}
        for idx in range(len(list_clients_X)):
            X=torch.Tensor(list_clients_X[idx])
            Y=torch.tensor(list_clients_y[idx])
            data[idx]=TensorDataset(X, Y)
            data_len[idx]=len(list_clients_y[idx])

        return data,data_len

