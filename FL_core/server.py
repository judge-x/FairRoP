import os.path

import torch
# import wandb
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import sys
import multiprocessing as mp
import random
import pandas as pd
from .client import Client
from .client_selection.config import *
from .pca import *
from .utils import *

# from ..model.ActorCritic import *


class Server(object):
    def __init__(self, data, init_model, args, selection, fed_algo, agent,poison_index):
        """
        Server to execute
        ---
        Args
            data: dataset for FL
            init_model: initial global model
            args: arguments for overall FL training
            selection: client selection method
            fed_algo: FL algorithm for aggregation at server
            results: results for recording
        """
        self.train_data = data['train']['data']
        self.train_sizes = data['train']['data_sizes']
        self.test_data = data['test']['data']
        self.test_sizes = data['test']['data_sizes']
        self.test_clients = data['test']['data_sizes'].keys()

        self.device = args.device
        self.args = args
        self.global_model = init_model
        self.selection_method = selection
        self.federated_method = fed_algo
        # self.files = files


        self.nCPU = mp.cpu_count() // 2 if args.nCPU is None else args.nCPU

        self.total_num_client = args.total_num_clients
        self.totol_client_index = [*range(self.total_num_client)]
        self.num_clients_per_round = args.num_clients_per_round
        self.num_available = args.num_available
        if self.num_available is not None:
            random.seed(args.seed)

        self.total_round = args.num_round
        self.save_results = not args.no_save_results
        self.save_probs = args.save_probs

        self.reward = 0
        self.R =0
        self.states=[]
        self.ac_train=args.ac_t

        self.per_selected_indices=[]
        self.poison_index=poison_index

        self.ep_greedy=args.ep_greedy

        # if self.save_probs:
        #     num_local_data = np.array([self.train_sizes[idx] for idx in range(args.total_num_client)])
        #     num_local_data.tofile(files['num_samples'], sep=',')
        #     files['num_samples'].close()
        #     del files['num_samples']

        self.test_on_training_data = False


        ## INITIALIZE
        # initialize the training status of each client
        self._init_clients(init_model)

        #initialize the multi-agents
        self.agent=agent

        # initialize the client selection method
        if self.args.method in NEED_SETUP_METHOD:
            self.selection_method.setup(self.train_sizes)

        if self.args.method in LOSS_THRESHOLD:
            self.ltr = 0.0



    def _init_clients(self, init_model):
        """
        initialize clients' model
        ---
        Args
            init_model: initial given global model
        """
        self.client_list = []
        self.Losses=[]
        flatten_models=[]
        if self.args.detect_or_not==1:
            for client_idx in range(self.total_num_client):
                local_train_data = self.train_data[client_idx]
                local_test_data = self.test_data[client_idx] if client_idx in self.test_clients else np.array([])
                model=deepcopy(init_model).to(self.device)
                flatten_models.append(flatten_full_model(model))
                c = Client(client_idx, self.train_sizes[client_idx], local_train_data, local_test_data,
                           model, self.args)
                self.client_list.append(c)
                # we set N as biggest norm of all updata
                N=100
                self.Losses.append(N)
                

            self.PCA_states=np.array(getPCA(flatten_models))
            print('get init pac successfully',len(self.PCA_states))
        else:
            for client_idx in range(self.total_num_client):
                local_train_data = self.train_data[client_idx]
                local_test_data = self.test_data[client_idx] if client_idx in self.test_clients else np.array([])
                model=deepcopy(init_model).to(self.device)
                c = Client(client_idx, self.train_sizes[client_idx], local_train_data, local_test_data,
                           model, self.args)
                self.client_list.append(c)
                # we set N as biggest norm of all updata
                N=100
                self.Losses.append(N)
    
        


    def train(self):
        """
        FL training
        """
        ## ITER COMMUNICATION ROUND
        Acc_train,Loss_train,local_losses=[],[],[]
        Acc_test=[]
        loss_vars, acc_vars = [], []
        Rewards=[]
        benign_index=[]
        poisoned_index_selected=[]
        count_selected_num=[0]*self.total_num_client

        for round_idx in range(self.total_round):
            print(f'\n>> ROUND {round_idx}')

            ## GET GLOBAL MODEL
            #self.global_model = self.trainer.get_model()
            self.global_model = self.global_model.to(self.device)

            # set clients
            client_indices = [*range(self.total_num_client)]

            if self.num_available is not None:
                print(f'> available clients {self.num_available}/{len(client_indices)}')
                np.random.seed(self.args.seed + round_idx)
                client_indices = np.random.choice(client_indices, self.num_available, replace=False)
                self.save_selected_clients(round_idx, client_indices)

            # set client selection methods
            # initialize selection methods by setting given global model
            if self.args.method in NEED_INIT_METHOD:
                local_models = [self.client_list[idx].trainer.get_model() for idx in client_indices]
                self.selection_method.init(self.global_model, local_models)
                del local_models

            # candidate client selection before local training
            if self.args.method in CANDIDATE_SELECTION_METHOD:
                # np.random.seed((self.args.seed+1)*10000000 + round_idx)
                print(f'> candidate client selection {self.args.num_candidates}/{len(client_indices)}')
                client_indices = self.selection_method.select_candidates(client_indices, self.args.num_candidates)


            ## PRE-CLIENT SELECTION
            # client selection before local training (for efficiency)
            self.per_selected_indices=[]
            if self.args.method in PRE_SELECTION_METHOD:
                if self.args.method == 'FairRoP':
                    selected_client_indices, exploration_flag=self.selection_method.select(round_idx)
                    print('Selected clients are:', selected_client_indices)
                else:
                    print(f'> pre-client selection {self.num_clients_per_round}/{len(client_indices)}')
                    client_indices = self.selection_method.select(self.num_clients_per_round, client_indices, None).tolist()
                    print("select index:", client_indices)

            ## CLIENT UPDATE (TRAINING)
            local_losses,accuracy, local_metrics= self.train_clients(client_indices,round_idx)

            local_models = [self.client_list[idx].trainer.get_model() for idx in self.totol_client_index]



            # Exploration
            if self.args.method == 'FairRoP':

                # update diff_norm
                # global_model_para=self.global_model.parameters()
                # global_para=torch.cat([param.view(-1) for param in self.global_model.parameters()])

                # selected_client_indices=self.selection_method.select(round_idx)

                # for idx in selected_client_indices:
                #     # print(idx)
                #     local_model=self.client_list[idx].trainer.get_model()
                #     local_para=torch.cat([param.view(-1) for param in local_model.parameters()]) 
                #     self.diff_norm[idx]=torch.norm(local_para-global_para, p=2).item()

                for idx in selected_client_indices:
                    self.Losses[idx]=local_losses[idx]

                # print("*******self_differ_norm********",self.diff_norm)
                # sample for fairness
                biggest_losses_index=[]
                Losses_= deepcopy(self.Losses)
                while len(biggest_losses_index)<len(selected_client_indices):
                    max=np.max(Losses_)
                    tmp=np.random.choice(np.flatnonzero(Losses_==max))
                    Losses_[tmp]=-1
                    biggest_losses_index.append(tmp)

                
                # if round_idx % 2 == 0:
                #     print("\nBiggest Loss Index:",biggest_losses_index)
                #     print(self.Losses)

                # updata PCA_states
                if self.args.detect_or_not == 1:
                    flatten_models=[]
                    for idx in selected_client_indices:
                        # flatten_models.append(flatten_models_gradient(self.global_model,self.client_list[idx].trainer.get_model()))
                        flatten_models.append(flatten_full_model(self.client_list[idx].trainer.get_model()))

                    weights_pac=getPCA(flatten_models)

                    for idx in range(len(selected_client_indices)):
                        self.PCA_states[selected_client_indices[idx]]=weights_pac[idx]

                    benign_index_selected,poison_index_detected, benign_index_detected=poison_detect(round_idx, self.PCA_states, selected_client_indices, self.poison_index)

                else:
                    benign_index_selected=selected_client_indices
                    benign_index_detected=client_indices
                    poison_index_detected=[]
                    
                
                self.selection_method.select_exploration(benign_index_detected, poison_index_detected, biggest_losses_index)
                # selected_client_indices=benign_index_selected

            ## CLIENT SELECTION
            if self.args.method not in PRE_SELECTION_METHOD:
                print(f'> post-client selection {self.num_clients_per_round}/{len(client_indices)}')
                kwargs = {'n': self.num_clients_per_round, 'client_idxs': client_indices, 'round': round_idx}
                kwargs['results'] = self.files['prob'] if self.save_probs else None
                # select by local models(gradients)
                if self.args.method in NEED_LOCAL_MODELS_METHOD:
                    local_models_ = [self.client_list[idx].trainer.get_model() for idx in client_indices]
                    selected_client_indices = self.selection_method.select(**kwargs, metric=local_models)
                    del local_models_
                # select by local losses
                elif self.args.method == 'FairRoP':
                    pass
                else:
                    selected_client_indices = self.selection_method.select(**kwargs, metric=local_metrics)
                if self.args.method in CLIENT_UPDATE_METHOD:
                    for idx in client_indices:
                        self.client_list[idx].update_ema_variables(round_idx)
                # update local metrics
                client_indices = np.take(client_indices, selected_client_indices).tolist()
                local_losses = np.take(local_losses, selected_client_indices)
                accuracy = np.take(accuracy, selected_client_indices)


            # count total selected numbers
            if self.args.method == 'FairRoP' or self.args.method =='DivFL':
                count_selected_num=count_client_num(count_selected_num,selected_client_indices)
            else:
                count_selected_num = count_client_num(count_selected_num, client_indices)

            acc_t,loss_t=self.save_current_updates(local_losses, accuracy, len(client_indices), phase='Train', round=round_idx)
            # self.save_selected_clients(round_idx, client_indices)

            #save training accuracy
            Acc_train.append(acc_t)
            Loss_train.append(loss_t)


            ## SERVER AGGREGATION
            # DEBUGGING
            # assert len(client_indices) == self.num_clients_per_round

            # aggregate local models
            if self.args.method=='FairRoP':
            # aggregate selected weights
                if self.args.fed_algo == 'FedAvg':
                    print('aggreate with Fedavg---FairRoP')
                    print('benign index:',benign_index_selected)
                    global_model_params= self.federated_method.update(local_models, benign_index_selected)
                elif self.args.fed_algo == 'q-ffl':
                    global_model_params = self.federated_method.update(self.global_model, local_models,
                                                                       benign_index_selected,local_losses,round_idx,self.args.qffl)
                else:
                    global_model_params = self.federated_method.update(local_models, benign_index_selected, self.global_model,
                                                                       self.client_list)
            else:
                if self.args.fed_algo == 'FedAvg':
                    global_model_params = self.federated_method.update(local_models, client_indices)
                elif self.args.fed_algo == 'q-ffl':
                    # global_model_params=self.global_model.state_dict()
                    global_model_params = self.federated_method.update(self.global_model, local_models,
                                                                       selected_client_indices,local_losses,round_idx,self.args.qffl)
                else:
                    global_model_params = self.federated_method.update(local_models, client_indices, self.global_model)



            # update aggregated model to global model
            self.global_model.load_state_dict(global_model_params)


            ## TEST
            if round_idx % self.args.test_freq == 0:
                self.global_model.eval()
                # test on train dataset
                # if self.test_on_training_data:
                #     acc_var,loss_var,acc,local_losses=self.test(self.total_num_client, phase='TrainALL')

                # test on benign dateset
                if self.args.poison_or_not==1:
                    clean_index=list(set(self.test_clients)-set(self.poison_index))
                    acc_var, loss_var, acc_test, loss_test=self.test_benign_client(clean_index,phase='Test')
                else:
                    acc_var,loss_var, acc_test, _ = self.test(len(self.test_clients), phase='Test')

                # acc_var,loss_var, acc_test, _ = self.test(len(self.test_clients), phase='Test')

                print("Accuracy_Variance:",acc_var, ' Loss Variance:', loss_var)

                acc_vars.append(acc_var)
                loss_vars.append(loss_var)
                Acc_test.append(acc_test)


                diff_acc = acc_test - Acc_test[len(Acc_test) - 1]

            del local_models, accuracy

        save_path="./"
        df1=pd.DataFrame({'Loss_vars':loss_vars,'Acc_vars':acc_vars,'Acc_train':Acc_train,'Loss_train':Loss_train,'Acc_test':Acc_test})
        df2=pd.DataFrame({'Count':count_selected_num})
        df3=pd.DataFrame({'Rewards':Rewards})
        df=pd.concat([df1,df2,df3],axis=1)
        df.to_excel(os.path.join(save_path,"result_{}_{}_{}_al{}_{}_{}.xlsx".format(self.args.method,self.args.dataset,self.args.poison_rate,self.args.dirichlet_alpha,self.args.posion_name,self.args.fed_algo)),index=False)
        return self.agent.model.state_dict()





    def local_training(self, client_idx, noise_attack):
        """
        train one client
        ---
        Args
            client_idx: client index for training
        Return
            result: trained model, (total) loss value, accuracy
        """
        client = self.client_list[client_idx]
        if self.args.method in LOSS_THRESHOLD:
            client.trainer.update_ltr(self.ltr)
        # result,flatted_weight,per_flatted_weight = client.train(deepcopy(self.global_model))
        # return result,flatted_weight,per_flatted_weight
        result = client.train(deepcopy(self.global_model),noise_attack)

        return result

    def local_testing(self, client_idx,test_on_training):
        """
        test one client
        ---
        Args
            client_idx: client index for test
            results: loss, acc, auc
        """
        client = self.client_list[client_idx]
        result = client.test(self.global_model, test_on_training)
        return result

    def train_clients(self, client_indices,round):
        """
        train multiple clients (w. or w.o. multi processing)
        ---
        Args
            client_indices: client indices for training
        Return
            trained models, loss values, accuracies
        """
        local_losses, accuracy, local_metrics = [], [], []

        local_loss_global,accuracy_global=[],[]
        ll, lh = np.inf, 0.
        # local training with multi processing
        flatten_weights=[]
        for client_idx in client_indices:
            if client_idx in self.poison_index and self.args.posion_name=='noise-attack':
                result = self.local_training(client_idx,True)
            else:
                result = self.local_training(client_idx,False)
            # result ,flatten_weight,per_flatten_weight= self.local_training(client_idx)

            local_losses.append(result['loss'])
            accuracy.append(result['acc'])
            local_metrics.append(result['metric'])

            if self.args.method in LOSS_THRESHOLD:
                if result['llow'] < ll: ll = result['llow'].item()
                lh += result['lhigh']

            progressBar(len(local_losses), len(client_indices), result)


            # per_weights_pac=[]


        if self.args.method in LOSS_THRESHOLD:
            lh /= len(client_indices)
            self.ltr = self.selection_method.update(lh, ll, self.ltr)

        return local_losses,accuracy,local_metrics

    def test_for_fix_client(self):
        loss_global,acc_global=[],[]

        for client_idx in range(10):
            result = self.local_testing(client_idx)

            loss_global.append(result['loss'])
            acc_global.append(result['acc'])
        return loss_global,acc_global

    def test(self, num_clients_for_test, phase):
        """
        test multiple clients
        ---
        Args
            num_clients_for_test: number of clients for test
            TrainALL: test on train dataset
            Test: test on test dataset
        """
        metrics = {'loss': [], 'acc': []}
        if phase=='Test':
            for client_idx in range(num_clients_for_test):
                result = self.local_testing(client_idx,True)

                metrics['loss'].append(result['loss'])
                metrics['acc'].append(result['acc'])

                progressBar(len(metrics['acc']), num_clients_for_test, result, phase='Test')
        else:
            for client_idx in range(num_clients_for_test,False):
                result = self.local_testing(client_idx)

                metrics['loss'].append(result['loss'])
                metrics['acc'].append(result['acc'])
                progressBar(len(metrics['acc']), num_clients_for_test, result)

        acc_var=get_variance([i for i in metrics['acc']])
        loss_var=get_variance([i for i in metrics['loss']])
        acc,loss=self.save_current_updates(metrics['loss'], metrics['acc'], num_clients_for_test, phase=phase)
        return acc_var,loss_var,acc,metrics['loss']

    def test_benign_client(self,client_index,phase='Test'):
        metrics = {'loss': [], 'acc': []}
        for client_idx in client_index:
            result = self.local_testing(client_idx,False)

            metrics['loss'].append(result['loss'])
            metrics['acc'].append(result['acc'])

            progressBar(len(metrics['acc']), len(client_index), result, phase='Test')

        acc_var=get_variance([i for i in metrics['acc']])
        loss_var=get_variance([i for i in metrics['loss']])
        acc,loss=self.save_current_updates(metrics['loss'], metrics['acc'], len(client_index), phase=phase)
        return acc_var,loss_var,acc,loss


    def save_current_updates(self, losses, accs, num_clients, phase='Train', round=None):
        """
        update current updated results for recording
        ---
        Args
            losses: losses
            accs: accuracies
            num_clients: number of clients
            phase: current phase (Train or TrainALL or Test)
            round: current round
        Return
            record "Round,TrainLoss,TrainAcc,TestLoss,TestAcc"
        """
        loss, acc = sum(losses) / num_clients, sum(accs) / num_clients

        if phase == 'Train':
            self.record = {}
            self.round = round
        self.record[f'{phase}/Loss'] = loss
        self.record[f'{phase}/Acc'] = acc
        status = num_clients if phase == 'Train' else 'ALL'

        print('> {} Clients {}ing: Loss {:.6f} Acc {:.4f}'.format(status, phase, loss, acc))

        if phase == 'Test':
            # wandb.log(self.record)
            if self.save_results:
                if self.test_on_training_data:
                    tmp = '{:.8f},{:.4f},'.format(self.record['TrainALL/Loss'], self.record['TrainALL/Acc'])
                else:
                    tmp = ''
                rec = '{},{:.8f},{:.4f},{}{:.8f},{:.4f}\n'.format(self.round,
                                                                  self.record['Train/Loss'], self.record['Train/Acc'],
                                                                  tmp,
                                                                  self.record['Test/Loss'], self.record['Test/Acc'])
                # self.files['result'].write(rec)
        return acc,loss

    def save_selected_clients(self, round_idx, client_indices):
        """
        save selected clients' indices
        ---
        Args
            round_idx: current round
            client_indices: clients' indices to save
        """
        self.files['client'].write(f'{round_idx + 1},')
        np.array(client_indices).astype(int).tofile(self.files['client'], sep=',')
        self.files['client'].write('\n')

    def weight_variance(self, local_models):
        """
        calculate the variances of model weights
        ---
        Args
            local_models: local clients' models
        """
        variance = 0
        for k in tqdm(local_models[0].state_dict().keys(), desc='>> compute weight variance'):
            tmp = []
            for local_model_param in local_models:
                tmp.extend(torch.flatten(local_model_param.cpu().state_dict()[k]).tolist())
            variance += torch.var(torch.tensor(tmp), dim=0)
        variance /= len(local_models)
        print('variance of model weights {:.8f}'.format(variance))

    def get_reward(self,acc,var1,var2):
        '''
        get reward by different points
        :param var:
        :return:
        '''
        reward_=-0.5*var1-0.5*var2
        # reward_=(pow(64,0.2*acc-0.8*100*var1))

        self.reward=reward_
        return reward_


