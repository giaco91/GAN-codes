# from __future__ import print_function
import matplotlib.pyplot as plt

import time
import os
import random
import numpy as np
from models.dc_generator import dc_generator
import torch
import torch.optim

from common_utils import *

#add data path
# import sys
# sys.path.append('/Users/Giaco/Documents/Elektrotechnik-Master/image_processing/my_deep_image_prior')

#---global settings

dtype = torch.FloatTensor

PLOT = True
TRAINING = True
imsize = 1024
load_model=True
num_iter = 100
show_every = 25
save_every=10
LR= 0.01
nz=2
ngf=6
add_noise=0
#----- specifiy the figure

random.seed(1)
torch.manual_seed(1)

img_path='photos/photos_corrupted/1.jpg'
mask_path='photos/photos_mask/0.jpg'

if not os.path.exists('saved_models/'):
    os.mkdir('saved_models')
if not os.path.exists('DIP_images/'):
    os.mkdir('DIP_images/')

#---- convert the images
#load real image
img_pil=square_crop(img_path)
img_pil=resize_to_height_ref(img_pil,imsize)
img_np=pil_to_np(img_pil)
print(img_np.shape)
img_np_array=np.zeros([1,3,imsize,imsize])
img_np_array[0,:]=img_np
img_torch_array=torch.from_numpy(img_np_array).type(dtype)
print('size of img_torch_array: '+str(img_torch_array.size()))

#load mask
mask_pil=square_crop(mask_path)
mask_pil=img_make_mask(mask_pil)
img_mask_pil=resize_to_height_ref(mask_pil,imsize)
img_mask_np = pil_to_np(img_mask_pil)
mask_torch = np_to_torch(img_mask_np).type(dtype)

# raise ValueError('')
#corrupt image
masked_images=img_torch_array*mask_torch

#---specify optimizer
pad = 'reflection' # 'zero'
OPTIMIZER = 'adam'

#-------image specific settigns----------
generator = dc_generator(nz,imgsize=imsize,ngf=ngf,nc=3)
        
#----torch inizializations
net_input=torch.randn(1, nz, 1, 1, device='cpu')
optimizer= torch.optim.Adam(generator.parameters(), lr=LR)
state_epoch=0
if load_model:
  print('reload model....')
  state_dict=torch.load('saved_models/generator_DIP.pkl')
  state_epoch=state_dict['epoch']
  generator.load_state_dict(state_dict['model_state'])
  optimizer.load_state_dict(state_dict['optimizer_state'])


# Compute number of parameters
s  = sum(np.prod(list(p.size())) for p in generator.parameters())
print ('Number of params: %d' % s)


def loss_function(inpainted,orig):
  if np.abs(add_noise)>0:
    de=inpainted-(orig+torch.randn(1, 3, imsize, imsize, device='cpu') * add_noise)
  else:
    de=inpainted-orig 
  loss=torch.sum(torch.mul(de,de))
  return loss

#training loop
i = state_epoch
def closure():    
    global i
    train_loss=0
    # batchSize=1
    optimizer.zero_grad()
    out = generator(net_input)
    loss=loss_function(out * mask_torch,masked_images)
    loss.backward()
    optimizer.step()
    train_loss+=loss.item()

    epoch_loss=train_loss
    print('Iteration: '+str(i)+'   '+'Loss: '+str(epoch_loss))
    if  PLOT and (i+1) % show_every == 0:
        print('save plot ...')
        out_np = torch.clamp(out[0,:].transpose(0,1).transpose(1,2).detach(),0,1).numpy()
        masked_np = masked_images[0,:].transpose(0,1).transpose(1,2).detach().numpy()
        orig_np=img_torch_array[0,:].transpose(0,1).transpose(1,2).detach().numpy()
        save_comparison_plot(masked_np,out_np,orig_np,'DIP_images/'+str(i))

    if (i+1)%save_every==0:
      print('save model ...')
      torch.save({'epoch': i, 'model_state': generator.state_dict(),'optimizer_state': optimizer.state_dict()}, 'saved_models/generator_DIP.pkl')
    i += 1

    return epoch_loss

#-----call optimizer and save stuff ----
if TRAINING:
  print('start training ...')
  for j in range(num_iter):
    closure()
  torch.save({'epoch': i, 'model_state': generator.state_dict(),'optimizer_state': optimizer.state_dict()}, 'saved_models/generator_DIP.pkl')

#save final result: take real image in the noncurrupted regions
reconstructed = generator(net_input)*(-mask_torch+1)+img_torch_array*mask_torch
reconstructed_np=reconstructed[0,:].transpose(0,1).transpose(1,2).detach().numpy()
masked_np = masked_images[0,:].transpose(0,1).transpose(1,2).detach().numpy()
orig_np=img_torch_array[0,:].transpose(0,1).transpose(1,2).detach().numpy()

save_comparison_plot(masked_np,reconstructed_np,orig_np,'DIP_images/final_reconstruction_'+str(i))






