# FairRoP
FairRoP:A fairness-aware federated client selection scheme with robust guarantee


## Base Requirements
```shell
torch=1.8.0
torchvision
numpy
scipy
tqdm
h5py
```

## Client Selection methods
```shell
python main.py --method {client selection method you want}
```

 1. ```,```: Random Selection
 3. ```Pow-d```: Power-of-d-Choice [[Yae Jee Cho et al., 2022](https://arxiv.org/pdf/2010.01243.pdf)]
 4. ```Cluster1```: Clustered Sampling 1 [[Yann Fraboni et al., 2021](http://proceedings.mlr.press/v139/fraboni21a/fraboni21a.pdf)]
 5. ```Cluster2```: Clustered Sampling 2 [[Yann Fraboni et al., 2021](http://proceedings.mlr.press/v139/fraboni21a/fraboni21a.pdf)]
 6. ```DivFL```: Diverse Client Selection for FL [[Ravikumar Balakrishnan et al., 2022](https://openreview.net/pdf?id=nwKXyFvaUm)]
 7. ```LTFCS/FairSec/FairRoP```: Ours. The LTFCS/FairSec is the synonms of FairRoP

## Benchmark Datasets

1. FederatedEMNIST (default)

   Download from this [[link](https://github.com/FedML-AI/FedML/blob/master/python/fedml/data/FederatedEMNIST/download_federatedEMNIST.sh)] and place them in your data directory ```data_dir```.
    
    ```shell
    python main.py --dataset FederatedEMNIST --model CNN -A 10 -K 50 --lr_local 0.1 -B 20 -R 100
   ```
   
2. FederatedEMNIST (non_IID/IID)

   ```shell
    python main.py --dataset FederatedEMNIST_nonIID/FederatedEMNIST_IID --model CNN -A 10 -K 50 --lr_local 0.1 -B 20 -R 100
   ```
   
3. FederatedEMNIST (Posioned without detech)

   3.1 Data poison
   ```shell
    python main.py --dataset FederatedEMNIST_Posi --model CNN -A 10 -K 50 --lr_local 0.1 -B 20 -R 100 -p 1 -p_n label-flipping -p_rt 0.25 -dete 0
   ```

   3.2 Model Poison

   
4. FederatedEMNIST (Posioned with detech)

   ```shell
    python main.py --dataset FederatedEMNIST_Posi --model CNN -A 10 -K 50 --lr_local 0.1 -B 20 -R 100 -eg 0.8 -p 1 -p_rt 0.25 -dete 1 --method LTFCS
   ```


5. FederatedCIFAR10 (Partitioned by Dirichlet distribution, followed by Clustered Sampling)
    
   ```shell
    python main.py --dataset Fed_CIFAR10_NIID -A 10 -K 50 --lr_local 0.1 -B 10 -R 150 --method FairRoP -al 1 -eg 0.5
   ```

6, More algorithm can be seen in ./utils/argparse.py


## References
 - https://github.com/euphoria0-0/Active-Client-Selection-for-Communication-efficient-Federated-Learning
