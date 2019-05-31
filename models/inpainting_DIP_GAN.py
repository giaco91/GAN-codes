# from __future__ import print_function
import matplotlib.pyplot as plt

import time
import os

import numpy as np
# from models.resnet import ResNet
# from models.unet import UNet
from models.skip import skip
import torch
import torch.optim

# from utils.inpainting_utils import *

#---global settings

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.FloatTensor

PLOT = True
TRAINING = False
imsize = 320
load_model=False
num_iter = 500
show_every = 1
save_every=1
max_num_img=10
batch_size=10
LR = 0.01

#----- specifiy the figure


#a low res rose image
img_path='data/rose/'
mask_path = 'data/inpainting/library_mask.png'


#specify the network architecture--

NET_TYPE = 'skip_depth4' # one of kip_depth6|skip_depth4|skip_depth2|UNET|ResNet

#------

#---- convert the images

img_list=[]
i=0
for img_name in os.listdir(img_path):
  if img_name.endswith(".jpg") and i<max_num_img:
    img_pil=square_crop(img_path+img_name)
    # print(img_pil.size)
    img_pil=resize_to_height_ref(img_pil,imsize)
    img_list.append(np.array(img_pil).astype(np.float32) / 255.)
    i+=1
n_img_loaded=len(img_list)
print('number of loaded images: '+str(n_img_loaded))
img_shape=img_list[0].shape
img_np_array=np.zeros([len(img_list),img_shape[0],img_shape[1],img_shape[2]])
for j in range(len(img_list)):
  img_np_array[j,:]=img_list[j]

img_torch_array=torch.from_numpy(img_np_array).type(dtype)


mask_pil=square_crop(mask_path)
img_mask_pil=resize_to_height_ref(mask_pil,imsize)
img_mask_np = pil_to_np(img_mask_pil)
mask_torch = np_to_torch(img_mask_np).type(dtype)

print('size of img_torch_array: '+str(img_torch_array.size()))

img_torch_array=torch.transpose(img_torch_array, 1, 3)
masked_images=img_torch_array*mask_torch


#---specify optimizer

pad = 'reflection' # 'zero'
OPTIMIZER = 'adam'


#-------image specific settigns----------

input_depth = 3
figsize = 5

   
depth = int(NET_TYPE[-1])
net = skip(input_depth, 3, 
       num_channels_down = [16, 32, 64, 128, 128, 128][:depth],
       num_channels_up =   [16, 32, 64, 128, 128, 128][:depth],
       num_channels_skip =    [0, 0, 0, 0, 0, 0][:depth],  
       filter_size_up = 3,filter_size_down = 5,  filter_skip_size=1,
       upsample_mode='nearest', # downsample_mode='avg',
       need1x1_up=False,
       need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
        
LR = 0.01 

#----torch inizializations

net_input=masked_images.detach().clone()

p = get_params('net', net, None)
optimizer= torch.optim.Adam(p, lr=LR)

if load_model:
  print('reload model....')
  state_dict=torch.load('saved_models/DIP_GAN_depth.pkl')
  state_epoch=state_dict['epoch']+0
  net.load_state_dict(state_dict['model_state'])
  optimizer.load_state_dict(state_dict['optimizer_state'])


# Compute number of parameters
s  = sum(np.prod(list(p.size())) for p in net.parameters())
print ('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)


#----- training loop--------
batch_idx=[0]
for b in range(int(np.floor(n_img_loaded/batch_size))):
  batch_idx.append((b+1)*batch_size)
if n_img_loaded>batch_size*(len(batch_idx)-1):
  batch_idx.append(n_img_loaded)

i = 0
def closure():
    
    global i
    train_loss=0
    for idx in range(len(batch_idx)-1):
      optimizer.zero_grad()

      # print('forward propagation ...')
      out = net(net_input[batch_idx[idx]:batch_idx[idx+1],:])
     
      # print('calc loss...')
      total_loss = mse(out * mask_torch, masked_images[batch_idx[idx]:batch_idx[idx+1],:])

      start_time = time.time()
      # print('back-prop....')
      total_loss.backward()
      train_loss+=total_loss.item()*(batch_idx[idx+1]-batch_idx[idx])
      # print("--- %s seconds ---" % (time.time() - start_time))
      optimizer.step()
        
    print('Iteration: '+str(i)+'   '+'Loss: '+str(train_loss/n_img_loaded))
    if  PLOT and i % show_every == 0:
        # n_im=i%n_img_loaded
        n_im=0
        print('save plot ...')
        out_np = out[n_im,:].transpose(0,1).transpose(1,2).detach().numpy()
        masked_np = masked_images[n_im,:].transpose(0,1).transpose(1,2).detach().numpy()
        orig_np=img_torch_array[n_im,:].transpose(0,1).transpose(1,2).detach().numpy()
        save_comparison_plot(masked_np,out_np,orig_np,'DIP-GAN_images/'+str(i)+'_'+str(n_im))

    if i%save_every==0:
      print('save model ...')
      torch.save({'epoch': i, 'model_state': net.state_dict(),'optimizer_state': optimizer.state_dict()}, 'saved_models/DIP_GAN_depth.pkl')

    i += 1

    return total_loss


#-----call optimizer and save stuff ----



#optimizer=optimize(OPTIMIZER, p, closure, LR, num_iter,optimizer=optimizer)
if TRAINING:
  print('start training ...')
  for j in range(num_iter):
    closure()
  torch.save({'epoch': num_iter, 'model_state': net.state_dict(),'optimizer_state': optimizer.state_dict()}, 'saved_models/DIP_GAN_depth.pkl')

#---- testing 
idx_1=0
idx_2=10
corrupted_img=net_input[idx_1:idx_2,:]
inpainted_img=net(corrupted_img)

corrupted_img = corrupted_img.transpose(1,2).transpose(2,3).detach().numpy()
inpainted_img = inpainted_img.transpose(1,2).transpose(2,3).detach().numpy()
original_img = img_torch_array[idx_1:idx_2,:].transpose(1,2).transpose(2,3).detach().numpy()

for j in range(corrupted_img.shape[0]):
  save_comparison_plot(corrupted_img[j,:],inpainted_img[j,:],original_img[j,:],'DIP-GAN_images/'+str(j))


