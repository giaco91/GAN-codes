import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

class Net1(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)#ouputsize (batchsize,6,28,28)
        self.pool = nn.MaxPool2d(2, 2)#outputsize (batchsize,6,14,14)
        self.conv2 = nn.Conv2d(6, 16, 5)#outputsize (batchsize,16,10,10)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 4,2,1)#ouputsize (batchsize,6,in/2,in/2)
        self.pool = nn.MaxPool2d(2, 2)#outputsize (batchsize,6,in/2,in/2)
        self.conv2 = nn.Conv2d(6, 16, 4,2,1)#outputsize (batchsize,16,in/2,in/2)
        self.fc1 = nn.Linear(16 * 2 * 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        a = F.relu(self.conv1(x))#32->16
        x = self.pool(a)#16->8
        a=F.relu(self.conv2(x))#8->4
        x = self.pool(a)#4->2
        x = x.view(-1, 16 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class list_net(nn.Module):
    def __init__(self):
        super(list_net, self).__init__()
        feature_extractor=[nn.ReflectionPad2d(1),
                 nn.Conv2d(3, 6, kernel_size=4, stride=2,padding=0, bias=True),
                 nn.LeakyReLU(0.2, inplace=True)]#32->16
        feature_extractor+=[nn.ReflectionPad2d(1),
                 nn.Conv2d(6, 16, kernel_size=4, stride=2,padding=0, bias=True),
                 nn.LeakyReLU(0.2, inplace=True)]#16->8
        self.conv_features = nn.Sequential(*feature_extractor)
        fc=[nn.Linear(16*8*8,120),
                  nn.LeakyReLU(0.2,inplace=True)]
        fc+=[nn.Linear(120,40),
                  nn.LeakyReLU(0.2,inplace=True)]
        fc+=[nn.Linear(40,10)]
        self.class_hypo = nn.Sequential(*fc)

    def forward(self, x):
        features=self.conv_features(x)
        features=features.view(features.size()[0],-1)
        return self.class_hypo(features)

class net_deep_conv(nn.Module):
    def __init__(self):
        super(net_deep_conv, self).__init__()
        feature_extractor=[nn.ReflectionPad2d(1),
                 nn.Conv2d(3, 6, kernel_size=4, stride=2,padding=0, bias=True),
                 nn.LeakyReLU(0.2, inplace=True)]#32->16
        feature_extractor+=[nn.ReflectionPad2d(1),
                 nn.Conv2d(6, 16, kernel_size=4, stride=2,padding=0, bias=True),
                 nn.LeakyReLU(0.2, inplace=True)]#16->8
        feature_extractor+=[nn.ReflectionPad2d(1),
                 nn.Conv2d(16, 32, kernel_size=4, stride=2,padding=0, bias=True),
                 nn.LeakyReLU(0.2, inplace=True)]#8->4
        feature_extractor+=[nn.ReflectionPad2d(1),
                 nn.Conv2d(32, 64, kernel_size=4, stride=2,padding=0, bias=True),
                 nn.LeakyReLU(0.2, inplace=True)]#4->2
        self.conv_features = nn.Sequential(*feature_extractor)
        fc=[nn.Linear(64*2*2,120),
                  nn.LeakyReLU(0.2,inplace=True)]
        fc+=[nn.Linear(120,10)]
        self.class_hypo = nn.Sequential(*fc)

    def forward(self, x):
        features=self.conv_features(x)
        features=features.view(features.size()[0],-1)
        return self.class_hypo(features)

class no_fc(nn.Module):
    def __init__(self):
        super(no_fc, self).__init__()
        feature_extractor=[nn.ReflectionPad2d(1),
                 nn.Conv2d(3, 6, kernel_size=4, stride=2,padding=0, bias=True),
                 nn.LeakyReLU(0.2, inplace=True)]#32->16
        feature_extractor+=[nn.ReflectionPad2d(1),
                 nn.Conv2d(6, 16, kernel_size=4, stride=2,padding=0, bias=True),
                 nn.LeakyReLU(0.2, inplace=True)]#16->8
        feature_extractor+=[nn.ReflectionPad2d(1),
                 nn.Conv2d(16, 32, kernel_size=4, stride=2,padding=0, bias=True),
                 nn.LeakyReLU(0.2, inplace=True)]#8->4
        feature_extractor+=[nn.Conv2d(32, 10, kernel_size=4, stride=1,padding=0, bias=True),
                 nn.LeakyReLU(0.2, inplace=True)]#4->1
        self.conv_features = nn.Sequential(*feature_extractor)

    def forward(self, x):
        features=self.conv_features(x)
        features=features.view(features.size()[0],-1)
        return features

class no_fc_with_bn(nn.Module):
    def __init__(self):
        super(no_fc_with_bn, self).__init__()
        feature_extractor=[nn.ReflectionPad2d(1),
                 nn.Conv2d(3, 6, kernel_size=4, stride=2,padding=0, bias=True),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.BatchNorm2d(6)]#32->16
        feature_extractor+=[nn.ReflectionPad2d(1),
                 nn.Conv2d(6, 16, kernel_size=4, stride=2,padding=0, bias=True),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.BatchNorm2d(16)]#16->8
        feature_extractor+=[nn.ReflectionPad2d(1),
                 nn.Conv2d(16, 32, kernel_size=4, stride=2,padding=0, bias=True),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.BatchNorm2d(32)]#8->4
        feature_extractor+=[nn.Conv2d(32, 10, kernel_size=4, stride=1,padding=0, bias=True),
                 nn.LeakyReLU(0.2, inplace=True)]#4->1
        self.conv_features = nn.Sequential(*feature_extractor)
        
    def forward(self, x):
        features=self.conv_features(x)
        features=features.view(features.size()[0],-1)
        return features

class dc_mult_sim(nn.Module):
    def __init__(self):
        super(dc_mult_sim, self).__init__()
        feature_extractor=[nn.ReflectionPad2d(1),
                 nn.Conv2d(3, 5, kernel_size=4, stride=2,padding=0, bias=True),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.BatchNorm2d(5)]#32->16
        feature_extractor+=[nn.ReflectionPad2d(1),
                 nn.Conv2d(5, 10, kernel_size=4, stride=2,padding=0, bias=True),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.BatchNorm2d(10)]#16->8
        feature_extractor+=[nn.ReflectionPad2d(1),
                 nn.Conv2d(10, 20, kernel_size=4, stride=2,padding=0, bias=True),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.BatchNorm2d(20)]#8->4
        feature_extractor+=[nn.ReflectionPad2d(1),
                  nn.Conv2d(20, 40, kernel_size=4, stride=2,padding=0, bias=True),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.BatchNorm2d(40)]#4->2
        feature_extractor+=[nn.Conv2d(40, 10, kernel_size=2, stride=1,padding=0, bias=True)]#2->1
        self.conv_features = nn.Sequential(*feature_extractor)
        
    def forward(self, x):
        features=self.conv_features(x)
        features=features.view(features.size()[0],-1)
        return features