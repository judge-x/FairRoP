from copy import deepcopy
from collections import OrderedDict
import torch
import numpy as np
import heapq
import copy
import scipy.stats as stats


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
            local_model = local_models[client_indices[idx]].cpu().state_dict()
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

class Ditto(FederatedAlgorithm):
    '''
    same as FedAvg
    '''
    def __init__(self, train_sizes, init_model):
        super().__init__(train_sizes, init_model)

    def update(self, local_models, client_indices, global_model):
        num_training_data = sum([self.train_sizes[idx] for idx in client_indices])
        update_model = OrderedDict()
        for idx in range(len(client_indices)):
            local_model = local_models[client_indices[idx]].cpu().state_dict()
            num_local_data = self.train_sizes[client_indices[idx]]
            weight = num_local_data / num_training_data
            for k in self.param_keys:
                if idx == 0:
                    update_model[k] = weight * local_model[k]
                else:
                    update_model[k] += weight * local_model[k]

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

    '''
    Bad Result
    
    '''
    def __init__(self, train_sizes, init_model, args):
        super().__init__(train_sizes, init_model)
        self.q=args.qffl
        self.args=args

    def update(self,global_model,local_models,client_indices,losses,round,q):

        update_model = OrderedDict()

        #calculate local hk
        hk = []
        L=1.0/self.args.lr_local

        norm_list=[]
        for item in local_models:
            # local_para=item.parameters()
            # global_para=global_model.parameters()

            local_para=torch.cat([param.view(-1) for param in item.parameters()])
            global_para=torch.cat([param.view(-1) for param in global_model.parameters()])
            diff=(local_para-global_para)*L
            # norm_list.append(torch.linalg.norm(torch.cat([p.view(-1) for p in item.parameters()-global_model.parameters()])))
            l2_norm = torch.norm(diff, p=2)  
            norm_list.append(l2_norm)

        # norm_list=self.gen_delta_norms(global_model,local_models)
        for idx in range(len(client_indices)):
            hk.append(q*pow(losses[idx],q-1)*norm_list[idx]+L*pow(losses[idx],q))

        global_model=global_model.state_dict()
        # updata omege
        if round!=0:
            for idx in range(len(client_indices)):

                local_model = local_models[idx].state_dict()
                for k in self.param_keys:
                    if idx == 0:
                        update_model[k] = (local_model[k]-global_model[k])
                    else:
                        update_model[k] += (local_model[k]-global_model[k])
        else:
            for idx in range(len(client_indices)):

                local_model = local_models[idx].state_dict()
                for k in self.param_keys:
                    if idx == 0:
                        update_model[k] = (local_model[k])
                    else:
                        update_model[k] += (local_model[k])


        # tmp=pow(losses[idx],q)*L
        # hk_=[hk[idx]/pow(losses[idx],q) for idx in range(len(hk))]
        hk_=[(1+q/(L*losses[idx])) for idx in range(len(losses))]
        hk_sum=sum(hk_)
        if round==0:
            for k in self.param_keys:
                # global_model[k]-=(update_model[k]/hk_sum).long()
                global_model[k]=update_model[k]/float(len(local_models))
        else:
            for k in self.param_keys:
                global_model[k]=global_model[k].float()-(update_model[k]/hk_sum)
                # global_model[k]-=(update_model[k]/10).long()

        return global_model
    
    # def gen_delta_norms(self, global_model, local_models):
    #     norm_list=[]
    #     for item in local_models:
    #         norm_list.append(torch.linalg.norm(torch.cat([p.view(-1) for p in item.parameters()-global_model.parameters()])))

    #     return norm_list


class afl(FederatedAlgorithm):
    '''
    not finish
    '''
    def __init__(self, train_sizes, init_model):
        super().__init__(train_sizes, init_model)
        self.lemma=[]

    def update(self, local_models, round,client_indices):
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
    


class FedKrum(FederatedAlgorithm):
    def __init__(self,train_sizes,init_model,args):
        super().__init__(train_sizes, init_model)
        self.num=args.total_num_clients
        self.p_rt=args.poison_rate

    '''
        update with multi-Krum
        refer to ''https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html''
    '''
    def update(self, local_models, client_indices,global_model):

        score=[]
        local_models_=[]
        for item in client_indices:
            local_models_.append(local_models[item])

        k=int((1-self.p_rt)*len(local_models_)-2)

        for idx in range(len(local_models_)):
            distance=0
            distance_=[]
            for idx_ in range(len(local_models_)):
                if idx_!=idx:
                    V=local_models_[idx].cpu().state_dict()
                    V_=local_models_[idx_].cpu().state_dict()
                    for key in self.param_keys:
                        distance += torch.dist(torch.tensor(V[key],dtype=torch.float), torch.tensor(V_[key],dtype=torch.float),p=2).item()
                    distance_.append(distance)

            minest_k=heapq.nsmallest(k,distance_)
            score_=sum(minest_k)
            score.append(score_)
        del V,V_,score_

        score_=score
        m=int(0.5*len(local_models_))
        smallest_m=heapq.nsmallest(m,score)
        sel_idx=[i for i, num in enumerate(score_) if num in smallest_m]
        print("the selected clients are:",sel_idx)
        num_training_data = sum([self.train_sizes[idx] for idx in sel_idx])
        update_model = OrderedDict()
        for i in range(len(sel_idx)):
            local_model = local_models_[sel_idx[i]].cpu().state_dict()
            num_local_data = self.train_sizes[sel_idx[i]]
            weight = num_local_data / num_training_data
            for k in self.param_keys:
                if i == 0:
                    update_model[k] = local_model[k]  * weight
                else:
                    update_model[k] += local_model[k] * weight

        return update_model


class FedPEFL(FederatedAlgorithm):
    def __init__(self, train_sizes, init_model,args):
        super().__init__(train_sizes, init_model)

    '''
    aggregred by PEFL with pearson relevation
    refer to ''https://ieeexplore.ieee.org/abstract/document/9524709''
    '''

    def update(self, local_models, client_indices,global_model):
        #get median
        local_models_=[]
        for item in client_indices:
            local_models_.append(local_models[item])
        local_models=local_models_


        median_model=OrderedDict()
        for k in self.param_keys:
            para_lists=[]
            for item in local_models:
                para_lists.append(item.cpu().state_dict()[k])

            model_stack=torch.stack(para_lists)
            median_model[k]=torch.median(model_stack,dim=0).values

        #get person relevent
        pere_rel=[]
        tensor1 = torch.cat([param.view(-1) for param in median_model.values()])
        for item in local_models:
            tensor2=torch.cat([param.view(-1) for param in item.cpu().state_dict().values()])
            corr,_=stats.pearsonr(np.array(tensor1),np.array(tensor2))
            pere_rel.append(corr)

        # num_training_data = sum([self.train_sizes[idx] for idx in client_indices])
        weight_pears=[]
        for item in pere_rel:
            # tmp=(1 + item) / (1 - item)
            weight_pears.append(max(0, np.log((1 + item) / (1 - item)) - 0.5))
        total_pear=sum(weight_pears)
        update_model = OrderedDict()
        for idx in range(len(client_indices)):
            local_model = local_models[idx].cpu().state_dict()
            # num_local_data = self.train_sizes[client_indices[idx]]

            weight = weight_pears[idx]/total_pear
            for k in self.param_keys:
                if idx == 0:
                    update_model[k] = weight * local_model[k]
                else:
                    update_model[k] += weight * local_model[k]

        return update_model

class FedBulyan(FederatedAlgorithm):
    def __init__(self,train_sizes,init_model,args):
        super().__init__(train_sizes, init_model)
        self.num=args.total_num_clients
        self.p_rt=args.poison_rate

    '''
        update with bulyan
        refer to ''http://proceedings.mlr.press/v80/mhamdi18a.html''
    '''
    def update(self, local_models, client_indices,global_model):

        local_models_=[]
        for item in client_indices:
            local_models_.append(local_models[item])
        local_models=local_models_

        num_bulyan=(1-2*self.p_rt)*len(local_models)
        init_idx=[i for i in range(len(local_models))]
        sel_idx=[]
        local_models_=copy.deepcopy(local_models)
        while len(sel_idx)<num_bulyan:
            k = int((1 - self.p_rt) * len(local_models_) - 2)
            score = []
            for idx in range(len(local_models_)):
                distance=0
                distance_=[]
                for idx_ in range(len(local_models_)):
                    if idx_!=idx:
                        V=local_models_[idx].cpu().state_dict()
                        V_=local_models_[idx_].cpu().state_dict()
                        for key in self.param_keys:
                            distance += torch.dist(torch.tensor(V[key],dtype=torch.float), torch.tensor(V_[key],dtype=torch.float),p=2).item()
                        distance_.append(distance)
                minest_k=heapq.nsmallest(k,distance_)
                score_=sum(minest_k)
                score.append(score_)
            sel_idx.append(init_idx[np.argmin(score)])
            init_idx.pop(np.argmin(score))
            local_models_.pop(np.argmin(score))

        sel_idx_true=[client_indices[i] for i in sel_idx]
        print("the selected clients are:",sel_idx_true)
        num_training_data = sum([self.train_sizes[idx] for idx in sel_idx])
        update_model = OrderedDict()
        for i in range(len(sel_idx)):
            local_model = local_models[sel_idx[i]].cpu().state_dict()
            num_local_data = self.train_sizes[sel_idx[i]]
            weight = num_local_data / num_training_data
            for k in self.param_keys:
                if i == 0:
                    update_model[k] = local_model[k] * weight
                else:
                    update_model[k] += local_model[k] * weight

        return update_model


class FedTrimmed_Mean(FederatedAlgorithm):
    def __init__(self,train_sizes,init_model,args):
        super().__init__(train_sizes, init_model)
        self.num=args.total_num_clients
        self.p_rt=args.poison_rate

    '''
        update with Fed Trimmed Mean
        refer to ''http://proceedings.mlr.press/v80/yin18a.html''
        目前还没有想好怎么排除掉模型每个维度上最大最小的k个点.  
        
    '''
    def update(self, local_models, client_indices,global_model):

        pass


class FedCluster(FederatedAlgorithm):
    def __init__(self, train_sizes, init_model, args):
        super().__init__(train_sizes, init_model)
        self.num = args.total_num_clients
        self.p_rt = args.poison_rate

    '''
        update with Fedcluster
        refer to ''Shielding Federated Learning: Robust Aggregation with Adaptive Client Selection''

    '''

    def update(self, local_models, client_indices, global_model):
        flattened_differ_models=[]
        for item in local_models:
            flattened_model = flatten_full_model(item)
            flattened_differ_models.append(flattened_model)

        models_pca=getPCA(flattened_differ_models)


        #clear outliner
        dis = []
        median = np.median(models_pca)
        model_pca_ = []
        outliner_index = []
        inliner_index = []
        for item in models_pca:
            dis.append(np.linalg.norm(item - median))
        mad = np.median(dis)
        for index in range(len(dis)):
            if dis[index] < (2 * mad):
                model_pca_.append(models_pca[index])
                inliner_index.append(index)
            else:
                outliner_index.append(index)
        pca_ = F.normalize(torch.FloatTensor(np.array(models_pca)))

        #cluster
        cls = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto', connectivity=None, linkage='ward',
                                      memory=None, n_clusters=2).fit(pca_)
        cls_cluster = cls.labels_.tolist()
        num_zero = cls_cluster.count(0)
        num_one = cls_cluster.count(1)

        if num_one > num_zero:
            clear_tag = 1
        else:
            clear_tag = 0

        selected_index=[]
        selected_index_true=[]
        for index in range(len(cls_cluster)):
            if cls_cluster[index] == clear_tag:
                selected_index.append(index)
                selected_index_true.append(client_indices[index])

        print('selected len is:',selected_index_true)



        num_training_data = sum([self.train_sizes[idx] for idx in selected_index])
        update_model = OrderedDict()
        for i in range(len(selected_index)):
            local_model = local_models[selected_index[i]].cpu().state_dict()
            num_local_data = self.train_sizes[selected_index[i]]
            weight = num_local_data / num_training_data
            for k in self.param_keys:
                if i == 0:
                    update_model[k] = local_model[k] * weight
                else:
                    update_model[k] += local_model[k] * weight


        return update_model