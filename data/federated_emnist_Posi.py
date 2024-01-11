import os
import random

import h5py
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset


# Date Poison Attack (Label-Fliapping) on neture dataset. Please first running federated_emnist.py


class FederatedEMNISTDataset_Posi:
    def __init__(self, data_dir, args):
        '''
        known class: digits (10)
        unknown class: characters (52) -> label noise
        '''
        self.num_classes = 10
        # self.train_num_clients = 3400 if args.total_num_clients is None else args.total_num_clients
        # self.test_num_clients = 3400 if args.total_num_clients is None else args.total_num_clients
        self.train_num_clients = args.total_num_clients
        self.test_num_clients = args.total_num_clients
        self.poison_rate = args.poison_rate
        self.poison_list=[]


        self._init_data(data_dir)
        print(f'Total number of users: {self.train_num_clients}')
        self.train_num_clients = len(self.dataset['train']['data_sizes'].keys())
        self.test_num_clients = len(self.dataset['test']['data_sizes'].keys())
        print(f'#TrainClients {self.train_num_clients} #TestClients {self.test_num_clients}')
        # 3383

    def _init_data(self, data_dir):
        file_name = os.path.join(data_dir, 'FederatedEMNIST_preprocessed_nonIID.pickle')

        print("Trying poison-attack.................")
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)
        else:
            dataset ,poison_indexs= preprocess(data_dir, self.train_num_clients, self.num_classes, self.poison_rate)

            # with open(file_name, 'wb') as f:
            #     pickle.dump(dataset, f)

        self.dataset = dataset
        self.poison_list=poison_indexs


def preprocess(data_dir, num_clients, num_classes, posi_rate):
    train_data = h5py.File(os.path.join(data_dir, 'FederatedEMNIST/fed_emnist_train.h5'), 'r')
    test_data = h5py.File(os.path.join(data_dir, 'FederatedEMNIST/fed_emnist_test.h5'), 'r')

    train_ids = list(train_data['examples'].keys())
    test_ids = list(test_data['examples'].keys())
    num_clients_train = len(train_ids) if num_clients is None else num_clients
    num_clients_test = len(test_ids) if num_clients is None else num_clients
    print(f'num_clients_train {num_clients_train} num_clients_test {num_clients_test}')

    # local dataset
    train_data_local_dict, train_data_local_num_dict = {}, {}
    test_data_local_dict, test_data_local_num_dict = {}, {}
    idx = 0

    # generate poisoned index
    poison_rate = posi_rate
    thr_index = int(num_clients_train * poison_rate)
    poison_list=random.sample([i for i in range(num_clients_train)],thr_index)

    for client_idx in range(num_clients_train):
        client_id = train_ids[client_idx]
        client_id2=train_ids[len(train_ids)-client_idx-1]
        # train
        tmp=train_data['examples'][client_id]['pixels'][()]
        train_x = np.expand_dims(np.concatenate((train_data['examples'][client_id]['pixels'][()],train_data['examples'][client_id2]['pixels'][()]),axis=0), axis=1)
        train_y = np.concatenate((train_data['examples'][client_id]['label'][()],train_data['examples'][client_id2]['label'][()]),axis=0)
        # set 36 classes as true labels
        # digits_index = np.arange(len(train_y))[np.isin(train_y, range(36))]

        # set the left 26 class as poisoned labels
        if client_idx in poison_list:
            non_digits_index = np.arange(len(train_y))[np.invert(np.isin(train_y, range(num_classes)))]
            train_y[non_digits_index] = np.random.randint(10, size=len(non_digits_index))

        if client_idx < 2000:
            # client with only digits
            train_y = train_y
            train_x = train_x

        else:
            # client with only characters (but it's label noise for digits classification)
            non_digits_index = np.invert(np.isin(train_y, range(10)))
            train_y = train_y[non_digits_index]
            train_y = np.random.randint(10, size=len(train_y))
            train_x = train_x[non_digits_index]

        if len(train_y) == 0:
            continue

        # test
        test_x = np.expand_dims(np.concatenate((test_data['examples'][client_id]['pixels'][()],test_data['examples'][client_id2]['pixels'][()]),axis=0), axis=1)
        test_y = np.concatenate((test_data['examples'][client_id]['label'][()],test_data['examples'][client_id2]['label'][()]),axis=0)

        if len(test_x) == 0:
            continue

        local_train_data = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
        train_data_local_dict[idx] = local_train_data
        train_data_local_num_dict[idx] = len(train_x)

        local_test_data = TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y))
        test_data_local_dict[idx] = local_test_data
        test_data_local_num_dict[idx] = len(test_x)

        idx += 1

    train_data.close()
    test_data.close()

    dataset = {}
    dataset['train'] = {
        'data_sizes': train_data_local_num_dict,
        'data': train_data_local_dict,
    }
    dataset['test'] = {
        'data_sizes': test_data_local_num_dict,
        'data': test_data_local_dict,
    }

    return dataset,poison_list