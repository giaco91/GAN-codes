# from __future__ import print_function
import matplotlib.pyplot as plt
import glob
import time
import os
import random
import numpy as np
from models.dc_generator import dc_generator
from models.dc_encoder import dc_encoder
from models.dc_discriminator import dc_discriminator
import torch
import torch.optim
import time

from utils.inpainting_utils import *

#---global settings

dtype = torch.FloatTensor
PLOT = True
TRAINING = True
train_disc=False
imsize = 64
load_model = True
num_iter =50
show_every = 25
shuffle=True
shuffle_every=1
save_every=20
max_num_img=80
batch_size=80#must be smaller or equal than max_num_img
LR_enc=LR_gen= 0.001
LR_disc=0.0001
latent_dim=5
l_reg_im=0#the influence of the discriminative regularization in the image space
l_reg_lat=0.1

mirror=True
rotate=True

nef=10
ngf=30
ndf=10
#----- specifiy the figure
random.seed(1)
torch.manual_seed(1)

# a low res rose image
# img_path='data/rose/'
# mask_path = 'data/inpainting/library_mask.png'
img_path='data/rose/'
mask_path='data/inpainting/library_mask.png'

#load images
img_list=[]
i=0
for img_name in os.listdir(img_path):
  if img_name.endswith(".jpg") and i<max_num_img:
    img_pil=square_crop(img_path+img_name)
    img_pil=resize_to_height_ref(img_pil,imsize)
    img_list.append(pil_to_np(img_pil))
    i+=1
    #----data augmentation-----
    if mirror and i<max_num_img:
      img_pil_mirror=PIL.ImageOps.mirror(img_pil)
      img_list.append(pil_to_np(img_pil_mirror))
      i+=1
      if rotate and i<max_num_img:
        img_list.append(pil_to_np(img_pil_mirror.rotate(90, expand=False)))
        i+=1
        if i<max_num_img:
          img_list.append(pil_to_np(img_pil_mirror.rotate(180, expand=False)))
          i+=1
        if i<max_num_img:
          img_list.append(pil_to_np(img_pil_mirror.rotate(270, expand=False)))
          i+=1
    if rotate and i<max_num_img:
      img_list.append(pil_to_np(img_pil.rotate(90, expand=False)))
      i+=1
      if i<max_num_img:
        img_list.append(pil_to_np(img_pil.rotate(180, expand=False)))
        i+=1
      if i<max_num_img:
        img_list.append(pil_to_np(img_pil.rotate(180, expand=False)))
        i+=1
      if i<max_num_img:
        img_list.append(pil_to_np(img_pil.rotate(270, expand=False)))
        i+=1

n_img_loaded=len(img_list)
print('number of loaded images: '+str(n_img_loaded))
img_shape=img_list[0].shape
img_np_array=np.zeros([len(img_list),img_shape[0],img_shape[1],img_shape[2]])
for j in range(len(img_list)):
  img_np_array[j,:]=img_list[j]
img_torch_array=torch.from_numpy(img_np_array).type(dtype)
# img_torch_array=torch.transpose(img_torch_array, 2, 3)
# img_torch_array=torch.transpose(img_torch_array, 1, 2)
print('size of img_torch_array: '+str(img_torch_array.size()))

#---load mask
masked_torch_array=torch.zeros(n_img_loaded,3,imsize,imsize)
for j in range(n_img_loaded):
  masked_torch_array[j,:,:,:]=torch.from_numpy(pil_to_np(get_mask(imsize,N=10,S=10))).type(dtype)


# mask_pil=square_crop(mask_path)
# mask_pil=img_make_mask(mask_pil)
# img_mask_pil=resize_to_height_ref(mask_pil,imsize)
# # img_show(img_mask_pil)
# img_mask_np = np.round(pil_to_np(img_mask_pil))
# mask_torch = np_to_torch(img_mask_np).type(dtype)

# masked_images=img_torch_array*mask_torch
masked_images=img_torch_array*masked_torch_array
neg_masked_images=img_torch_array.detach().clone().masked_fill_((masked_images*-1+1).type(torch.ByteTensor), -1)

# test_mask_pil=get_mask(imsize)
# test_mask_pil.show()
# PIL.ImageOps.mirror(test_mask_pil).show()
# print(pil_to_np(test_mask_pil).shape)
# test_mask_pil_np=np.array(test_mask_pil).astype(np.float32) / 255.
# print(test_mask_pil_np.shape)

# raise ValueError('alles wird gut!')

#----torch inizializations
net_input=neg_masked_images.detach().clone()
print('inputshape '+str(net_input.size()))
#net_input=torch.randn(n_img_loaded, latent_dim, 1, 1, device='cpu')
#print(net_input[0,0,0,0])
#print('net input size '+str(net_input.size()))
#---specify optimizer
OPTIMIZER = 'adam'

if not os.path.exists('saved_models/'):
    os.mkdir('saved_models')
if not os.path.exists('conv_ae_images_latent_disc/') and PLOT:
    os.mkdir('conv_ae_images_latent_disc')
#-------image specific settigns----------
encoder = dc_encoder(latent_dim,imsize,ndf=nef)
generator = dc_generator(latent_dim,ngf=ngf,nc=3,imgsize=imsize)
discriminator = dc_discriminator(imsize,ndf=ndf,nc=3,max_depth=7)

optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=LR_enc)
optimizer_gen= torch.optim.Adam(generator.parameters(), lr=LR_gen)
optimizer_disc= torch.optim.Adam(discriminator.parameters(), lr=LR_disc)
state_epoch=0
if load_model:
  print('reload model....')
  state_dict_enc=torch.load('saved_models/conv_ae_enc_latent_disc'+str(imsize)+'.pkl')
  state_dict_gen=torch.load('saved_models/conv_ae_gen_latent_disc'+str(imsize)+'.pkl')
  state_dict_disc=torch.load('saved_models/conv_ae_disc_latent_disc'+str(imsize)+'.pkl')
  state_epoch=state_dict_gen['epoch']
  encoder.load_state_dict(state_dict_enc['model_state'])
  generator.load_state_dict(state_dict_gen['model_state'])
  discriminator.load_state_dict(state_dict_disc['model_state'])
  optimizer_enc.load_state_dict(state_dict_enc['optimizer_state'])
  optimizer_gen.load_state_dict(state_dict_gen['optimizer_state'])
  optimizer_disc.load_state_dict(state_dict_disc['optimizer_state'])


# Compute number of parameters
np_enc = sum(np.prod(list(p.size())) for p in encoder.parameters())
np_gen  = sum(np.prod(list(p.size())) for p in generator.parameters())
np_tot=np_enc+np_gen
print ('Number of params in dc_autoencoder: %d' % np_tot)

def disc_loss_fuction(d,d_hat,batchSize):
  print('detect reals with prob: '+str(torch.sum(d)/batchSize))
  print('detect fakes with prob: '+str(torch.sum(1-d_hat)/batchSize))
  loss=torch.sum(-torch.log(d)-torch.log(1-d_hat))
  return loss/batchSize

def ae_loss_function(inpainted,orig,latent_out,d_hat,batchSize):
  de=inpainted-orig 
  loss=torch.sum(torch.mul(de,de))
  latent_reg=l_reg_im*torch.sum(-torch.log(d_hat))
  # print('regularization loss: '+str(latent_reg/batchSize))
  loss+=latent_reg
  return loss/batchSize

#----- training loop--------
batch_idx=[0]

for b in range(int(np.floor(n_img_loaded/batch_size))):
  batch_idx.append((b+1)*batch_size)
if n_img_loaded>batch_size*(len(batch_idx)-1):
  batch_idx.append(n_img_loaded)


i = state_epoch
def closure():  
    global i
    global net_input
    global img_torch_array
    train_loss=0
    if shuffle and (i+1)%shuffle_every==0:
      shuffle_idx=torch.randperm(net_input.size()[0])
      net_input=img_torch_array.detach().clone().masked_fill_((-img_torch_array*masked_torch_array[shuffle_idx]+1).type(torch.ByteTensor), -1)[shuffle_idx]
      img_torch_array=img_torch_array[shuffle_idx]

    for idx in range(len(batch_idx)-1):
      batchSize=batch_idx[idx+1]-batch_idx[idx]
      optimizer_gen.zero_grad()
      optimizer_enc.zero_grad()
      optimizer_disc.zero_grad()

#-----propagate the autoencoder

      latent_out=encoder(net_input[batch_idx[idx]:batch_idx[idx+1],:]) 
      out = generator(latent_out)
      latent_std=torch.sqrt(torch.sum(torch.mul(latent_out,latent_out))/batchSize)
      print('standard deviation in latent_space: '+str(latent_std))
 
      if train_disc:
#----discriminator training
        d=discriminator(img_torch_array[batch_idx[idx]:batch_idx[idx+1],:])
        d_hat = discriminator(out.detach())
        disc_loss = disc_loss_fuction(d,d_hat,batchSize)
        disc_loss.backward()
        optimizer_disc.step()

      d_hat=discriminator(out)
      # ae_loss=ae_loss_function(out * mask_torch,masked_images[batch_idx[idx]:batch_idx[idx+1],:],latent_out,d_hat,batchSize)
      ae_loss=ae_loss_function(out,img_torch_array[batch_idx[idx]:batch_idx[idx+1],:],latent_out,d_hat,batchSize)
      ae_loss.backward()
      optimizer_enc.step()
      optimizer_gen.step()
      train_loss+=ae_loss.item()

    epoch_loss=train_loss
    print('Iteration: '+str(i)+'   '+'Loss: '+str(epoch_loss/imsize**2))
    if  PLOT and (i+1) % show_every == 0:
        # n_im=i%n_img_loaded
        print('save plot ...')
        for n_im in range(np.minimum(10,batch_size)):
          out_np = torch.clamp(out[n_im,:].transpose(0,1).transpose(1,2).detach(),0,1).numpy()
          masked_np = torch.clamp(net_input[+n_im,:].transpose(0,1).transpose(1,2).detach(),0,1).numpy()
          orig_np=img_torch_array[n_im,:].transpose(0,1).transpose(1,2).detach().numpy()
          save_comparison_plot(masked_np,out_np,orig_np,'conv_ae_images_latent_disc/'+str(i)+'_'+str(n_im))

    if (i+1)%save_every==0:
      print('save model ...')
      torch.save({'epoch': i, 'model_state': encoder.state_dict(),'optimizer_state': optimizer_enc.state_dict()}, 'saved_models/conv_ae_enc_latent_disc_'+str(imsize)+'.pkl')
      torch.save({'epoch': i, 'model_state': generator.state_dict(),'optimizer_state': optimizer_gen.state_dict()}, 'saved_models/conv_ae_gen_latent_disc_'+str(imsize)+'.pkl')  
      torch.save({'epoch': i, 'model_state': discriminator.state_dict(),'optimizer_state': optimizer_disc.state_dict()}, 'saved_models/conv_ae_disc_latent_disc'+str(imsize)+'.pkl')    
    i += 1

    return epoch_loss

#-----call optimizer and save stuff ----
if TRAINING:
  print('start training ...')
  for j in range(num_iter):
    closure()
  torch.save({'epoch': i, 'model_state': generator.state_dict(),'optimizer_state': optimizer_gen.state_dict()}, 'saved_models/conv_ae_gen_latent_disc'+str(imsize)+'.pkl')
  torch.save({'epoch': i, 'model_state': encoder.state_dict(),'optimizer_state': optimizer_enc.state_dict()}, 'saved_models/conv_ae_enc_latent_disc'+str(imsize)+'.pkl')
  torch.save({'epoch': i, 'model_state': discriminator.state_dict(),'optimizer_state': optimizer_disc.state_dict()}, 'saved_models/conv_ae_disc_latent_disc'+str(imsize)+'.pkl') 
#---- testing 

# corrupted_img=net_input+(-2+j*0.1)
if PLOT:
  N=20
  corrupted_img=torch.randn(N, latent_dim, 1, 1, device='cpu')
  with torch.no_grad():
    latent_out=encoder(net_input[0:100,:])
    latent_std=torch.sqrt(torch.sum(torch.mul(latent_out,latent_out))/100)
    print('standard deviation in latent_space: '+str(latent_std))
    corrupted_img=torch.randn(N, latent_dim, 1, 1, device='cpu')*latent_std
    inpainted_img=torch.clamp(generator(corrupted_img), 0, 1).transpose(1,2).transpose(2,3).detach().numpy()
  original_img = img_torch_array[0:1,:].transpose(1,2).transpose(2,3).detach().numpy()[0,:]
  corrupted_img = masked_images[0:1,:].transpose(1,2).transpose(2,3).detach().numpy()[0,:]
  for j in range(N):
    inpainted_img_j = inpainted_img[j:j+1,:]
    save_comparison_plot(corrupted_img[0,:],inpainted_img_j[0,:],original_img[0,:],'conv_ae_images_latent_disc/'+str(j))



