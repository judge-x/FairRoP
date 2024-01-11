'''
    Long Term fair client selection


'''
import torch
from scipy.stats import beta
import numpy as np
from .client_selection import ClientSelection
import torch.nn.functional as F

class FairRoP(ClientSelection):
    def __init__(self, total, device,eplison_greedy,num_clients_per_round):
        super().__init__(total, device)

        self.eplison_greedy=eplison_greedy
        self.client_weights=self.init_client_weight(total)
        self.per_select_num_client=num_clients_per_round


    def init_client_weight(self,total_num_client):
        return [[1,1] for i in range(total_num_client)]

    def updata_weight(self,selected_index):
        fair=1
        robust=1

        for i in range(self.total):
            if i in selected_index:
                self.client_weights[i][1]+=fair
            else:
                self.client_weights[i][0]+=robust

    def update_weight_norm(self,last_n_index,benign_index,posoined_index):
        # loss_all=F.normalize(torch.tensor(loss_all,dtype=torch.float64),dim=0).tolist()
        # mean=np.mean(loss_all)
        # for i in range(len(loss_all)):
        #     loss_all[i]-=mean
        fair=1
        robust=2

        for i in range(self.total):
            if i in benign_index:
                if i in last_n_index:
                    self.client_weights[i][0]+=fair
            if i in posoined_index:
                self.client_weights[i][1]+=robust

    def get_beta_goal(self):
        goal=[]
        x=np.arange(0,1,0.005)
        for key,var in enumerate(self.client_weights):
            goal.append(np.random.beta(var[0], var[1]))
            # goal.append(np.argmax(beta.pdf(x, var[0], var[1])))

        return goal

    def eplsion_sampling(self,client_goals):
        '''
        return the pre-selected client index Beta sampling
        :param client_goals:
        :return:
        '''
        result_clients=[]
        # exporation or explicaion
        count = 0
        while count<self.per_select_num_client:
            max=np.max(client_goals)
            tmp=np.random.choice(np.flatnonzero(client_goals==max))
            client_goals[tmp]=-1
            result_clients.append(tmp)
            count+=1

        return result_clients


    def select(self, round_idx):
        # exploration deceay
        if round_idx%10==0:
            self.eplison_greedy*=0.9

        exploration_flag = True if np.random.uniform() <= self.eplison_greedy else False

        if exploration_flag:
            print('\nExporation：')
            # random select the explore clients
            eplison_selected_client=self.first_select()

        else:
            #get importance selected clients
            print('\nExploitation：')

            beta_goals=self.get_beta_goal()

            eplison_selected_client=self.eplsion_sampling(beta_goals)

        return eplison_selected_client,exploration_flag
    
    def select_exploration(self, benign_index, posioned_index, last_norm_index):

        self.update_weight_norm(last_norm_index, benign_index, posioned_index)

    def select_exploitation(self):

        beta_goals=self.get_beta_goal()

        exploit_selected_client=self.eplsion_sampling(beta_goals)

        return exploit_selected_client

    def first_select(self):
        return np.random.choice(np.arange(self.total),self.per_select_num_client,replace=False).tolist()


    def count(pre_count,selected_index):
        for i in range(len(selected_index)):
            pre_count[selected_index[i]]+=1

        return pre_count


if __name__=='__main__':
    kwargs={'total':50,'device':'cpu'}
    lt=LTFCS(**kwargs, eplison_greedy=0.2,num_clients_per_round=5)

    a=np.arange(20)
    last_selected=np.random.choice(a,10,replace=False)
    print(last_selected)

    count_num=[0]*20

    for i in range(10):
        selected=lt.select(last_selected)
        count_num=count(count_num,selected)
        last_selected=selected
        # print(selected)
    print(count_num)
