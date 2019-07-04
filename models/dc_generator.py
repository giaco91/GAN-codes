import os
import torch
import torch.nn as nn
import numpy as np

class dc_generator(nn.Module):
    def __init__(self,nz,ngf=10,nc=3,imgsize=256,sigmoid=True,max_n_channel=10000):
        super(dc_generator, self).__init__()
        self.need_sigmoid=sigmoid
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
        if out_ch>max_n_channel:
            out_ch=max_n_channel
        print(out_ch)
        self.deconv.add_module("layer"+str(1),nn.Sequential(nn.ConvTranspose2d(nz, out_ch, 2,1, 0, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.BatchNorm2d(out_ch))
        )
        for i in range(1,self.depth-1):
            # in_ch=int(ngf*np.power(2,self.depth-1-i))
            in_ch=out_ch
            out_ch=int(ngf*np.power(2,self.depth-2-i))
            if out_ch>max_n_channel:
                out_ch=max_n_channel
            print(out_ch)
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
                nn.BatchNorm2d(nc)))
            else:
                self.deconv.add_module("layer"+str(self.depth),nn.Sequential(nn.ConvTranspose2d(in_ch, nc, 4,2, (missing+1)/2+1, output_padding=1, bias=False)))     
        else:
            self.deconv.add_module("layer"+str(self.depth),nn.Sequential(nn.ConvTranspose2d(in_ch, nc, 4,2, 1, bias=False))
            )
        self.sigmoid=nn.Sigmoid()


    def forward(self, input):
        output = self.deconv(input)
        if self.need_sigmoid:
            output=self.sigmoid(output)
        return output

class dc_root_generator(nn.Module):
    def __init__(self,nz,ngf=10,nc=3,imgsize=256,sigmoid=True,max_n_channel=10000):
        super(dc_root_generator, self).__init__()
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
        if out_ch>max_n_channel:
            out_ch=max_n_channel
        print(out_ch)
        self.deconv.add_module("layer"+str(1),nn.Sequential(nn.ConvTranspose2d(nz, out_ch, 2,1, 0, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.BatchNorm2d(out_ch))
        )
        for i in range(1,self.depth-1):
            in_ch=out_ch
            out_ch=int(ngf*np.power(2,self.depth-2-i))
            if out_ch>max_n_channel:
                out_ch=max_n_channel
            print(out_ch)
            self.deconv.add_module("layer"+str(i+1),nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, 4,2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_ch))
            )
        in_ch=ngf
        self.in_ch=in_ch
        self.output_padding=0
        if self.rest>0:
            missing=int(np.power(2,self.depth)-imgsize)
            if missing%2==0:
                self.input_padding=missing/2+1
            else:
                self.input_padding=(missing+1)/2+1
                self.output_padding=1  
        else:
            self.input_padding=1

    def forward(self, input):
        output = self.deconv(input)
        return output

class dc_final_generator(nn.Module):
    def __init__(self,in_ch,input_padding=1,output_padding=0,nc=3,imgsize=256,sigmoid=True):
        super(dc_final_generator, self).__init__()
        self.final = nn.Sequential(nn.ConvTranspose2d(in_ch, nc, 4,2, padding=input_padding, output_padding=output_padding, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(nc))
        self.sigmoid=nn.Sigmoid()
        self.need_sigmoid=sigmoid

    def forward(self, input):
        output = self.final(input)
        if self.need_sigmoid:
            output=self.sigmoid(output)
        return output

