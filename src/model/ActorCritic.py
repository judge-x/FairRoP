import torch
from torch import nn
import torch.nn.functional as F


class ac_net(nn.Module):
    def __init__(self,n_states,n_actions):
        super(ac_net, self).__init__()
        self.n_states=n_states
        self.n_actions=n_actions

        self.action_layer=nn.Sequential(
            nn.Linear(self.n_states,128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128,n_actions),
            nn.Softmax(dim=1)
        )

        self.value_layer = nn.Sequential(
            nn.Linear(self.n_states, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    
    def forward(self,x):
        actions=self.action_layer(x)
        value=self.value_layer(x)
        return actions,value

class Agent(nn.Module):
    def __init__(self,n_per_clients,gramma,lr):
        super(Agent, self).__init__()
        self.n_action=n_per_clients
        self.gramma=gramma
        self.lr=lr

        self.model=ac_net(n_per_clients*2,self.n_action)

        # print(self.model)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.lr,eps=1e-05)


    def threshold(self,probs):

        # print(probs)
        # print(torch.mean(probs))
        return torch.where(probs>=torch.mean(probs),torch.tensor(1),torch.tensor(0))

    def choose_action(self,s):
        s=torch.unsqueeze(torch.FloatTensor(s),0)
        s=F.normalize(s)
        probs,_=self.model(s)
        # print(probs)
        action=self.threshold(probs).numpy().tolist()
        indexs=[]
        for item in range(len(action[0])):
            if action[0][item]==1:
                indexs.append(item)

        return indexs,probs.squeeze(0)

    def critic_learn(self,s,s_next,reward):
        s=torch.unsqueeze(torch.FloatTensor(s),0)
        s_next=torch.unsqueeze(torch.FloatTensor(s_next),0)

        s = F.normalize(s)
        s_next= F.normalize(s_next)

        _,v=self.model(s)
        _,v_next=self.model(s_next)


        target=reward+self.gramma*v_next

        loss_func=nn.MSELoss()
        loss=loss_func(v,target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=0.5)
        self.optimizer.step()

        advantage=(target-v).detach()

        return advantage

    def actor_learn(self,s,advantage,indexs):
        _,probs=self.choose_action(s)
        loss_probs = probs.log()
        loss_prob=0
        for item in indexs:
            loss_prob+=loss_probs[item]

        loss_prob=loss_prob/len(indexs)
        loss=-advantage*loss_prob

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
