'''
Reference:
    FedML: https://github.com/FedML-AI/FedML
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


'''Federated EMNIST'''
class CNN_DropOut(nn.Module):
    def __init__(self, only_digits=False, num_channel=1):
        super(CNN_DropOut, self).__init__()
        self.conv2d_1 = nn.Conv2d(num_channel, 32, kernel_size=3)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout_1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(9216, 128)
        self.dropout_2 = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(128, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.dropout_1(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        x = self.linear_2(x)
        #x = self.softmax(self.linear_2(x))
        return x

# class CNN_EMNIST(nn.Module):
#     def __init__(self, only_digits=False, num_channel=1):
#         super(CNN_EMNIST, self).__init__()
#         self.conv2d_1 = nn.Conv2d(num_channel, 32, kernel_size=3)
#         self.max_pooling = nn.MaxPool2d(2, stride=2)
#         self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=3)
#         self.flatten = nn.Flatten()
#         self.linear_1 = nn.Linear(9216, 128)
#         self.linear_2 = nn.Linear(128, 10 if only_digits else 62)
#         self.relu = nn.ReLU()
#         #self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         x = self.conv2d_1(x)
#         x = self.relu(x)
#         x = self.conv2d_2(x)
#         x = self.relu(x)
#         x = self.max_pooling(x)
#         x = self.flatten(x)
#         x = self.linear_1(x)
#         x = self.relu(x)
#         x = self.linear_2(x)
#         #x = self.softmax(self.linear_2(x))
#         return x

class CNN_EMNIST(nn.Module):

    def __init__(self):
        super(CNN_EMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 62)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.pool(x)
        x = self.pool(self.batchnorm2(F.relu(self.conv2(x))))
        x = self.pool(self.batchnorm3(F.relu(self.conv3(x))))
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

'''CelebA'''
class CNN(nn.Module):
    def __init__(self, num_channel=3, num_class=2):
        super(CNN, self).__init__()
        self.layer1 = self._make_layer(num_channel, 32, 15)
        self.layer2 = self._make_layer(32, 32, 15)
        self.layer3 = self._make_layer(32, 32, 16)
        self.layer4 = self._make_layer(32, 32, 16)
        self.fc = nn.Linear(1152, num_class)

        self.layer1.apply(xavier_uniform)
        self.layer2.apply(xavier_uniform)
        self.layer3.apply(xavier_uniform)
        self.layer4.apply(xavier_uniform)
        self.fc.apply(xavier_uniform)


    def _make_layer(self, inp, outp=32, pad=0):
        layers = [
            nn.Conv2d(inp, outp, kernel_size=3, padding=(outp - inp)//2),
            nn.BatchNorm2d(outp),
            nn.MaxPool2d(outp, stride=2, padding=pad),
            nn.ReLU()
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



def xavier_uniform(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


'''CelebA

Reference:
    https://github.com/PengchaoHan/EasyFL
'''

class ModelCNNCeleba(nn.Module):
    def __init__(self):
        super(ModelCNNCeleba, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      ),

            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      ),

            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      ),

            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      ),

            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Linear(1152, 2)

    def forward(self, x):
        output1 = self.conv1(x)
        output2 = self.conv2(output1)
        output3 = self.conv3(output2)
        output4 = self.conv4(output3)
        output = output4.view(-1, 1152)
        output = self.fc(output)
        return output



'''Partitioned CIFAR100'''
class CNN_CIFAR_dropout(torch.nn.Module):
    """Model Used by the paper introducing FedAvg"""

    def __init__(self, num_class=10):
        super(CNN_CIFAR_dropout, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=(3, 3)
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3)
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3)
        )

        self.fc1 = nn.Linear(4 * 4 * 64, 64)
        self.fc2 = nn.Linear(64, num_class)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.dropout(x)
        x = x.view(-1, 4 * 4 * 64)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return x


# class CNN_CIFAR(nn.Module):
#     """from torch tutorial

#         https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#     """
#     def __init__(self):
#         super(CNN_CIFAR,self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1)  # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    
class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# class CNN_CIFAR(nn.Module):

#     def __init__(self):
#         super(CNN_CIFAR, self).__init__()

#         self.conv_layer = nn.Sequential(

#             # Conv Layer block 1
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # Conv Layer block 2
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout2d(p=0.05),

#             # Conv Layer block 3
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )

#         self.fc_layer = nn.Sequential(
#             nn.Dropout(p=0.1),
#             nn.Linear(4096, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.1),
#             nn.Linear(512, 10)
#         )

    # def forward(self, x):
    #     """Perform forward."""

    #     # conv layers
    #     x = self.conv_layer(x)

    #     # flatten
    #     x = x.view(x.size(0), -1)

    #     # fc layer
    #     x = self.fc_layer(x)

    #     return x

class AlexNet_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet_CIFAR10, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


'''CNN For FMNIST'''
class CNN_FMNIST(nn.Module):
    def __init__(self):
        super(CNN_FMNIST, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    


class ResNet9(nn.Module):
    def __init__(self, in_channels=3, dim=512):
        super().__init__()
        self.prep = conv_bn_relu_pool(in_channels, 64)
        self.layer1_head = conv_bn_relu_pool(64, 128, pool=True)
        self.layer1_residual = nn.Sequential(conv_bn_relu_pool(128, 128), conv_bn_relu_pool(128, 128))
        self.layer2 = conv_bn_relu_pool(128, 256, pool=True)
        self.layer3_head = conv_bn_relu_pool(256, 512, pool=True)
        self.layer3_residual = nn.Sequential(conv_bn_relu_pool(512, 512), conv_bn_relu_pool(512, 512))
        self.MaxPool2d = nn.Sequential(
            nn.MaxPool2d(4))
        self.linear = nn.Linear(dim, 10)
        # self.classifier = nn.Sequential(
        #     nn.MaxPool2d(4),
        #     nn.Flatten(),
        #     nn.Linear(512, num_classes))


    def forward(self, x):
        x = self.prep(x)
        x = self.layer1_head(x)
        x = self.layer1_residual(x) + x
        x = self.layer2(x)
        x = self.layer3_head(x)
        x = self.layer3_residual(x) + x
        x = self.MaxPool2d(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.linear(x)
        return x
    
def conv_bn_relu_pool(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        #nn.GroupNorm(32,out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)