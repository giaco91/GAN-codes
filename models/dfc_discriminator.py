import torch
import torch.nn as nn

class dfc_discriminator(nn.Module):
    def __init__(self,input_dim,layer_size=[10,10,10]):
        super(dfc_discriminator, self).__init__()
        self.dfc_discr = nn.Sequential()
        for i in range(len(layer_size)):
            if i==0:
                in_dim=input_dim
            else:
                in_dim=layer_size[i-1]
            self.dfc_discr.add_module("layer"+str(i+1),nn.Sequential(nn.Linear(in_dim, layer_size[i]),
            nn.LeakyReLU()))
        self.dfc_discr.add_module("layer"+str(len(layer_size)+1),nn.Sequential(nn.Linear(layer_size[-1], 1),
            nn.Sigmoid()))

    def forward(self, input):
        return self.dfc_discr(input)


