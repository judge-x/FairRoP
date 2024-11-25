# Title: FairRoP: Robust Client Selection Scheme for Fairness-aware Federated Learning

## System: 
Our code is available: https://github.com/judge-x/FairRoP/tree/main. 
The core code follow that:https://github.com/judge-x/FairRoP/blob/main/FL_core/client_selection/fairrop.py and https://github.com/judge-x/FairRoP/blob/main/FL_core/server.py

## Parepare Dataset: Our datasets are available online.
FMNIST: Supports two download methods: 1. Automatic download using torchvision, 2. Manual download for links
```
1, torchvision.datasets.FashionMNIST(os.path.join(data_dir, 'FMNIST'), train=train, download=True)
2, resources = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz"
http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz"
"http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz"
"http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"]
```
EMNIST: Download for links:
```
https://github.com/FedML-AI/FedML/blob/master/python/fedml/data/FederatedEMNIST/download_federatedEMNIST.sh
```
CIFAR10: Like FMNIST
```
1, torchvision.datasets.CIFAR10(os.path.join(data_dir,'PartitionedCIFAR10'), train=train, transform=transform_aug, download=True,)
2, https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

## Environment: Create a conda virtual environment directly from environment.yaml
(https://github.com/judge-x/FairRoP/blob/main/environment.yaml)
```
conda env create -f environment.yaml
```
**Base Requirements**
torch=1.8.0
torchvision
numpy
scipy
tqdm
h5py

## Benchmarks:
**Client Selection:**
1. ```,```: Random Selection
2. ```Pow-d```: Power-of-d-Choice [[Yae Jee Cho et al., 2022](https://arxiv.org/pdf/2010.01243.pdf)]
3. ```DivFL```: Diverse Client Selection for FL [[Ravikumar Balakrishnan et al., 2022](https://openreview.net/pdf?id=nwKXyFvaUm)]

**Robust:**
 1. ```FedKrum```: (https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html)
 2. ```Bylyan```: (http://proceedings.mlr.press/v80/mhamdi18a.html')
 3. ```FEFL```: (https://ieeexplore.ieee.org/abstract/document/9524709)
 4. ```Ditto```: (https://proceedings.mlr.press/v139/li21h/li21h.pdf)

## Running Code:
**Common Shell:**
python main.py --dataset {--args dataset} --model {--args model} -A {--args participted clients} -K {--args total clients} --lr_local {--args learning rate} -B {--args local batchsize} -R {--args global round} -al {--args NIID Rate} -eg {--args init exploartion rate} -p {--args poison or not} -p {--args attack methods} -p_rt {--args poison rate} -dete {--args detect or not} --method {--args client selection methods} --fel_algo {--args aggregation algorithm}

**FMNIST**
```
# On IID FMNIST
python main.py --dataset Fed_EMNIST_IID --model CNN -A 10 -K 50 --lr_local 0.1 -B 20 -R 100

# On poisoned/NIID FMNIST with detech
python main.py --dataset Fed_FMNIST_NIID_POSI --model CNN -A 10 -K 50 --lr_local 0.1 -B 20 -R 100 -p 1 -p_rt 0.2 -dete 1 -eg 0.8 --method FairRoP -al 0.5
```

**EMNIST**
```
# On IID EMNIST
python main.py --dataset Fed_FMNIST_IID --model CNN -A 10 -K 50 --lr_local 0.1 -B 20 -R 100

# On poisoned/NIID EMNIST with detech
python main.py --dataset Fed_EMNIST_NIID_POSI --model CNN -A 10 -K 50 --lr_local 0.1 -B 20 -R 100 -p 1 -p_rt 0.2 -dete 1 -eg 0.8 --method FairRoP -al 0.5
```

**CIFAR10**
```
# On poisoned/NIID CIFAR10 with detech
python main.py --dataset Fed_CIFAR10_POSI --model CNN -A 10 -K 50 --lr_local 0.1 -B 10 -R 150 -p 1 -p_rt 0.2 -dete 1 -eg 0.8 --method FairRoP -al 0.5
```

## Import argument
```
# trianing dataset
'--dataset':choices=['Fed_EMNIST', 'Fed_CIFAR100', 'Fed_CIFAR10_NIID', 'Fed_CIFAR10_POSI', 'Fed_EMNIST_IID', 'Fed_EMNIST_NIID', 'Fed_EMNIST_POSI', 'Fed_EMNIST_NIID_POSI','Fed_FMNIST_NIID','Fed_FMNIST_NIID_POSI']) 

# heterogeneous rate
'-al', default=0.5

# robust aggregation algorithm
'--fed_algo', default='FedAvg',choices = ['FedAvg','FedKrum','FedPEFL','FedBulyan','Ditto']

# methods
'--method', default= 'FairRoP', choices = ['Pow-d','DivFL','Random', 'FairRoP', 'Fix']

# init exploration rate
'-eg', default= 0.8
```
