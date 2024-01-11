import sys
import numpy as np
import torch
import math
from scipy.stats import stats
from scipy import spatial
from sklearn.cluster import AgglomerativeClustering
import torch.nn.functional as F
from collections import Counter
from .pca import drawPAC_detected

def progressBar(idx, total, result, phase='Train', bar_length=20):
    """
    progress bar
    ---
    Args
        idx: current client index or number of trained clients till now
        total: total number of clients
        phase: Train or Test
        bar_length: length of progress bar
    """
    percent = float(idx) / total
    arrow = '=' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r> Client {}ing: [{}] {}% ({}/{}) Loss {:.6f} Acc {:.4f}".format(
        phase, arrow + spaces, int(round(percent * 100)), idx, total, result['loss'], result['acc'])
    )
    sys.stdout.flush()

def get_variance(list):

    return np.var(list)*10000

def get_states(states ,index):
    selected_states=[]
    for item in index:
        selected_states.append(states[item * 2])
        selected_states.append(states[item * 2 + 1])

    return selected_states

def count_client_num(count_list,client_indexs):

    for item in client_indexs:
        count_list[item]+=1

    return count_list

def Gaussion_noise(grad_tensor, sigma):
    # sensitvte=2*C/bitch_size
    # c=math.sqrt(2*math.log(1.25/delta))
    # sigma=c*sensitvte/eplsion
    # sigma=0.025

    gaussion_noise=torch.normal(0,sigma,grad_tensor.shape).cuda()

    return grad_tensor+gaussion_noise

def poison_detect_(weight_pca,seleced_index,poison_index):
    '''
    detect by median
    '''

    medium=np.median(weight_pca,axis=0)

    #get person
    person_coefficient=[]
    for item in weight_pca:
        person_coefficient.append(1-spatial.distance.cosine(item,medium))

    bengin_index=[]
    posioned_index=[]
    for item in seleced_index:
        tmp=recale_fun(person_coefficient[item])
        if(tmp==0):
            posioned_index.append(item)
        else:
            bengin_index.append(item)

    test_detect(seleced_index, poison_index, bengin_index)

    return bengin_index,posioned_index

def poison_detect(round_idx, weight_pca,seleced_index,poison_index):
    '''
    detect by cluster
    '''
    # weight_pca=list(filter(lambda x: (x[0]!=0 and x[1]!=0),weight_pca))
    existed_weight_pca=[]
    existed_index=[]
    for idx, weight in enumerate(weight_pca):
        if weight[0]!=0 and weight[1]!=0:
            existed_index.append(idx)
            existed_weight_pca.append(weight)

    weight_pca_,inliner_index,poison_index_detected=clear_outliers(existed_weight_pca, existed_index)

    benign_index_detected=[]

    # cluster 
    K=2

    cls = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto', connectivity=None, linkage='ward',
                                  memory=None, n_clusters=K).fit(weight_pca_)
    cls_cluster=cls.labels_.tolist()


    # k cluster:
    cnt=Counter(cls_cluster)
    num_each_cluster=[]
    for key,value in cnt.items():
        num_each_cluster.append(value)
    
    thr_var=10
    dis=cluster_distance(weight_pca_,cls_cluster)
    print("dis is :",dis)
    if dis>thr_var:
        clean_tag=Counter(cls_cluster).most_common(K)[0][0]


        # two cluster :
        # num_zero=cls_cluster.count(0)
        # num_one=cls_cluster.count(1)

        # if num_one>num_zero:
        #     poison_tag=1
        # else:
        #     poison_tag=0

        for index in range(len(cls_cluster)):
            if cls_cluster[index]!=clean_tag:
                poison_index_detected.append(inliner_index[index])
            else:
                benign_index_detected.append(inliner_index[index])
    else:
        benign_index_detected=inliner_index

    bengin_index_select=list(set(seleced_index)-set(poison_index_detected))
    poison_index_select=list(set(seleced_index)-set(bengin_index_select))

    if len(bengin_index_select)==0:
        bengin_index_select=list(set(seleced_index) & set(inliner_index))

    test_detect(seleced_index, poison_index, poison_index_select)

    if round_idx%25==0:
        drawPAC_detected(weight_pca, benign_index_detected, poison_index_detected, round_idx)

    return bengin_index_select, poison_index_detected, benign_index_detected



def clear_outliers(weight_pca,seleced_index):
    '''
    Clear the outliner with MAD
    '''
    dis=[]
    median=np.median(weight_pca,axis=0)
    pca_=[]
    outliner_index=[]
    inliner_index=[]

    for item in weight_pca:
        dis.append(np.linalg.norm(item - median))
    mad = np.median(dis)

    for index in range(len(dis)):
        if dis[index] < (2 * mad):
            pca_.append(weight_pca[index])
            inliner_index.append(seleced_index[index])
        else:
            outliner_index.append(seleced_index[index])

    # pca_=F.normalize(torch.FloatTensor(np.array(pca_)))


    return pca_,inliner_index,outliner_index


def cluster_distance(weight_pca, cls_cluster):
    '''
    calculate the distance of two cluster
    '''

    cluster0, cluster1=[], []

    for idx in range(len(cls_cluster)):
        if cls_cluster[idx]==0:
            cluster0.append(weight_pca[idx])
        else:
            cluster1.append(weight_pca[idx])

    cluster0_mean=np.mean(cluster0,axis=0)
    cluster1_mean=np.mean(cluster1,axis=0)
    dis=np.linalg.norm(cluster1_mean-cluster0_mean)

    return dis

def recale_fun(p):

    return max(0,math.log((1+p)/(1-p))-0.5)

def test_detect(selected_index,poison_index,poison_index_select):
    a = list(set(selected_index) & set(poison_index))
    b = list(set(poison_index) & set(poison_index_select))
    print()
    if len(a)!=0 :
        # print('posioned rate in selected clients is:',a/len(selected_index))
        print("poison index in select is :" ,a , "detect index in select is :", poison_index_select, 'detect rate:', len(b)/len(a))
    else:
        print('no posioned attacker this round')

def get_last_index(losses,n=10):

    losses_tmp=losses.copy()
    result=[]
    i=0
    while i<n:
        tmp=np.argmax(losses_tmp)
        losses_tmp[tmp]=-1
        result.append(tmp)
        i+=1
    return result
