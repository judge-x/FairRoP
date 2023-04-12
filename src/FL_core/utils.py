import sys
import numpy as np
import torch
import math
from scipy.stats import stats
from scipy import spatial
from sklearn.cluster import AgglomerativeClustering
import torch.nn.functional as F

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

    return np.var(list)

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

def Gaussion_noise(grad_tensor,eplsion,delta,C,bitch_size):
    sensitvte=2*C/bitch_size
    c=math.sqrt(2*math.log(1.25/delta))
    sigma=c*sensitvte/eplsion
    # sigma=0.025

    gaussion_noise=torch.normal(0,sigma,grad_tensor.shape).cuda()

    return grad_tensor+gaussion_noise

def poison_detech_(weight_pca,seleced_index,poison_index):
    '''
    detech by median
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

    test_detech(seleced_index, poison_index, bengin_index)

    return bengin_index,posioned_index

def poison_detech(weight_pca,seleced_index,poison_index):
    '''
    detech by cluster
    '''

    weight_pca_,inliner_index,outliner_index=clear_outliers(weight_pca)
    cls = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto', connectivity=None, linkage='ward',
                                  memory=None, n_clusters=2).fit(weight_pca_)
    cls_cluster=cls.labels_.tolist()
    num_zero=cls_cluster.count(0)
    num_one=cls_cluster.count(1)

    if num_one>num_zero:
        clear_tag=1
    else:
        clear_tag=0

    for index in range(len(cls_cluster)):
        if cls_cluster[index]!=clear_tag:
            outliner_index.append(inliner_index[index])

    seleced_index_=list(set(seleced_index)-set(outliner_index))

    test_detech(seleced_index, poison_index, seleced_index_)
    return seleced_index_,outliner_index



def clear_outliers(weight_pca):
    dis=[]
    median=np.median(weight_pca)
    pca_=[]
    outliner_index=[]
    inliner_index=[]
    for item in weight_pca:
        dis.append(np.linalg.norm(item - median))
    mad = np.median(dis)
    for index in range(len(dis)):
        if dis[index] < (2 * mad):
            pca_.append(weight_pca[index])
            inliner_index.append(index)
        else:
            outliner_index.append(index)
    pca_=F.normalize(torch.FloatTensor(np.array(pca_)))
    return pca_,inliner_index,outliner_index

def recale_fun(p):

    return max(0,math.log((1+p)/(1-p))-0.5)

def test_detech(selected_index,poison_index,detech_index):
    a=len(set(selected_index)&set(poison_index))
    b=len(set(poison_index)&set(detech_index))
    print()
    if a!=0 :
        print('posioned rate in selected clients is:',a/len(selected_index))
        print('posioned rate afer detech:', b/a)
    else:
        print('no posioned attacker')

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
