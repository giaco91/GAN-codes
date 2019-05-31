import os
import torch
import torch.nn as nn
import numpy as np

class dc_generator(nn.Module):
    def __init__(self,nz,ngf=10,nc=3,imgsize=256):
        super(dc_generator, self).__init__()
        depth=int(np.floor(np.log2(imgsize)))
        self.rest=imgsize-np.power(2,depth)
        if self.rest>0:
            print('generator: image size is not power of two')
            self.depth=int(depth+1)
        else:
            self.depth=int(depth)
        print('initialized a dc generator with depth = '+str(self.depth))
        self.deconv = nn.Sequential()
        out_ch=int(ngf*np.power(2,self.depth-2))
        self.deconv.add_module("layer"+str(1),nn.Sequential(nn.ConvTranspose2d(nz, out_ch, 2,1, 0, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.BatchNorm2d(out_ch))
        )
        for i in range(1,self.depth-1):
            in_ch=int(ngf*np.power(2,self.depth-1-i))
            out_ch=int(ngf*np.power(2,self.depth-2-i))
            self.deconv.add_module("layer"+str(i+1),nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, 4,2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_ch))
            )
        in_ch=ngf
        if self.rest>0:
            missing=int(np.power(2,self.depth)-imgsize)
            if missing%2==0:
                self.deconv.add_module("layer"+str(self.depth),nn.Sequential(nn.ConvTranspose2d(in_ch, nc, 4,2, missing/2+1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(nc),nn.Sigmoid())
                
                )
            else:
                self.deconv.add_module("layer"+str(self.depth),nn.Sequential(nn.ConvTranspose2d(in_ch, nc, 4,2, (missing+1)/2+1, output_padding=1, bias=False),
                nn.Sigmoid())
                
                )     
        else:
            self.deconv.add_module("layer"+str(self.depth),nn.Sequential(nn.ConvTranspose2d(in_ch, nc, 4,2, 1, bias=False),
            nn.Sigmoid())
            )


    def forward(self, input):
        output = self.deconv(input)
        return output

