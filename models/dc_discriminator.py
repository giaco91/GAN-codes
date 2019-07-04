import torch
import torch.nn as nn
import numpy as np

class dc_discriminator(nn.Module):
    def __init__(self,imgsize,ndf,nc=3,max_depth=7):
        super(dc_discriminator, self).__init__()
            #nc: Number of channels in the training images. For color images this is 3
            #ndf: # Size of feature maps
            # input is (nc) x imgsize x imgsize
            # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            # ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
        depth=int(np.minimum(np.floor(np.log2(imgsize)),max_depth-1))
        self.rest=imgsize/np.power(2,depth)
        if self.rest>1:
            print('discriminator: image size is not power of two or max_depth is quite small. Imgsize: '+str(imgsize))
            self.depth=int(depth+1)
        else:
            self.depth=int(depth)
        print('initialized a dc discriminator with depth = '+str(self.depth))
        self.dc_discr = nn.Sequential()
        for i in range(self.depth-1):
            if i==0:
                in_ch=nc
            else:
                in_ch=int(ndf*np.power(2,i-1))
            out_ch=int(ndf*np.power(2,i))
            self.dc_discr.add_module("layer"+str(i+1),nn.Sequential(nn.Conv2d(in_ch, out_ch, 4,2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_ch))
            )
            # state size. (ndf*2**i x imgsize/(2**(i+1)) x imgsize/(2**(i+1))
        if self.rest>1:
            self.dc_discr.add_module("layer"+str(self.depth),nn.Sequential(nn.Conv2d(int(ndf*np.power(2,self.depth-2)), 1, self.rest, 1, 0, bias=False),
            nn.Sigmoid())
            )
        else:
            self.dc_discr.add_module("layer"+str(self.depth),nn.Sequential(nn.Conv2d(int(ndf*np.power(2,self.depth-2)), 1, 2, 1, 0, bias=False),
            nn.Sigmoid())
            )


    def forward(self, input):
        return self.dc_discr(input).view(-1, 1).squeeze(1)


