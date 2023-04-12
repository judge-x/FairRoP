from copy import deepcopy
from collections import OrderedDict
import torch
import numpy as np


class FederatedAlgorithm:
    def __init__(self, train_sizes, init_model):
        self.train_sizes = train_sizes
        if type(init_model) == OrderedDict:
            self.param_keys = init_model.keys()
        else:
            self.param_keys = init_model.cpu().state_dict().keys()

    def update(self, local_models, client_indices, global_model=None):
        pass



class FedAvg(FederatedAlgorithm):
    def __init__(self, train_sizes, init_model):
        super().__init__(train_sizes, init_model)

    def update(self, local_models, client_indices):
        num_training_data = sum([self.train_sizes[idx] for idx in client_indices])
        update_model = OrderedDict()
        for idx in range(len(client_indices)):
            local_model = local_models[idx].cpu().state_dict()
            num_local_data = self.train_sizes[client_indices[idx]]
            weight = num_local_data / num_training_data
            for k in self.param_keys:
                if idx == 0:
                    update_model[k] = weight * local_model[k]
                else:
                    update_model[k] += weight * local_model[k]

        return update_model

    def update_poison(self, local_models, client_indices,poison_scale):
        num_training_data = sum([self.train_sizes[idx] for idx in client_indices])
        update_model = OrderedDict()
        print(poison_scale)
        for idx in range(len(client_indices)):
            local_model = local_models[idx].cpu().state_dict()
            num_local_data = self.train_sizes[client_indices[idx]]
            weight = num_local_data / num_training_data
            for k in self.param_keys:
                if idx == 0:
                    update_model[k] =  local_model[k] * poison_scale[idx] * weight
                else:
                    update_model[k] += local_model[k] * poison_scale[idx] * weight

        return update_model


class FedAdam(FederatedAlgorithm):
    def __init__(self, train_sizes, init_model, args):
        super().__init__(train_sizes, init_model)
        self.beta1 = args.beta1  # 0.9
        self.beta2 = args.beta2  # 0.999
        self.epsilon = args.epsilon  # 1e-8
        self.lr_global = args.lr_global
        self.m, self.v = OrderedDict(), OrderedDict()
        for k in self.param_keys:
            self.m[k], self.v[k] = 0., 0.

    def update(self, local_models, client_indices, global_model):
        num_training_data = sum([self.train_sizes[idx] for idx in client_indices])
        gradient_update = OrderedDict()
        for idx in range(len(local_models)):
            local_model = local_models[idx].cpu().state_dict()
            num_local_data = self.train_sizes[client_indices[idx]]
            weight = num_local_data / num_training_data
            for k in self.param_keys:
                if idx == 0:
                    gradient_update[k] = weight * local_model[k]
                else:
                    gradient_update[k] += weight * local_model[k]
                torch.cuda.empty_cache()

        global_model = global_model.cpu().state_dict()
        update_model = OrderedDict()
        for k in self.param_keys:
            g = gradient_update[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * torch.mul(g, g)
            m_hat = self.m[k] / (1 - self.beta1)
            v_hat = self.v[k] / (1 - self.beta2)
            update_model[k] = global_model[k] - self.lr_global * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return update_model

class qFFL(FederatedAlgorithm):
    def __init__(self, train_sizes, init_model, args):
        super().__init__(train_sizes, init_model)
        self.q=args.qffl

    def update(self,global_model,local_models,client_indices,losses,lr,q):
        global_model=global_model.state_dict()
        update_model = OrderedDict()
        hk = 0
        for idx in range(len(client_indices)):

            F = np.float_power(losses[idx]*10 + 1e-10, q)
            norm = torch.linalg.norm(torch.cat([p.view(-1) for p in local_models[idx].parameters()]))
            hk+=q*np.float_power(losses[idx]*10+1e-10,q-1)*norm+F/lr

            local_model = local_models[idx].state_dict()
            for k in self.param_keys:
                if idx == 0:
                    update_model[k] = (local_model[k]-global_model[k])/lr*F
                else:
                    update_model[k] += (local_model[k]-global_model[k])/lr*F


        for k in self.param_keys:
            global_model[k]-=(update_model[k]/hk).long()

        return global_model