import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
import torch.nn.functional as F
import os.path

def flatten_full_model(model):
    model=torch.nn.utils.parameters_to_vector(model.parameters()).cpu().data.numpy()
    return np.asarray(model)

def flatten_models_gradient(global_model,local_model):
    global_model=torch.nn.utils.parameters_to_vector(global_model.parameters()).cpu().data.numpy()
    local_model=torch.nn.utils.parameters_to_vector(local_model.parameters()).cpu().data.numpy()

    different_model=global_model-local_model
    return np.asarray(different_model)

def getPCA(flatted_models):
    flatted_models=StandardScaler().fit_transform(flatted_models)
    nan_clear=SimpleImputer(strategy='mean')
    flatted_models=nan_clear.fit_transform(flatted_models)
    # print(flatted_models)
    pca=PCA(n_components=2)
    principalCompents=pca.fit_transform(flatted_models)

    return principalCompents

def drawPAC(principalComopents,posi_list):
    color = ['red', 'black','blue']
    # principalDf=pd.DataFrame(data=principalComopents,columns=['c1','c2'])

    #get coordinate-wise medians
    # principalComopents=F.normalize(torch.FloatTensor(principalComopents))
    median=np.median(principalComopents,axis=0)


    for i in range(len(principalComopents)):
        if i in posi_list:
            plt.scatter(principalComopents[i][0], principalComopents[i][1], color=color[1])
        else:
            plt.scatter(principalComopents[i][0], principalComopents[i][1], color=color[0])
    plt.scatter(median[0],median[1],color=color[2])
    plt.show()

def drawPAC_detected(weight_pca, beinign_index, pois_index, round_idx):
    color = ['green','red']
    # principalDf=pd.DataFrame(data=principalComopents,columns=['c1','c2'])

    #get coordinate-wise medians
    # principalComopents=F.normalize(torch.FloatTensor(principalComopents))
    # median=np.median(principalComopents,axis=0)


    for i in range(len(weight_pca)):
        if i in pois_index:
            plt.scatter(weight_pca[i][0], weight_pca[i][1], color=color[1])
        if i in beinign_index:
            plt.scatter(weight_pca[i][0], weight_pca[i][1], color=color[0])
    # plt.scatter(median[0],median[1],color=color[2])
    plt.show()
    path='./pca_image'
    plt.savefig(os.path.join(path,"{}.png".format(round_idx)))




