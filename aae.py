# from __future__ import print_function
import matplotlib.pyplot as plt
import glob
import time
import os
import random
import numpy as np
from models.dc_generator import dc_generator
from models.dc_encoder import dc_encoder
from models.dfc_discriminator import dfc_discriminator
import torch
import torch.optim
import time

from common_utils import *

#---global settings

dtype = torch.FloatTensor
PLOT = True
TRAINING = True
imsize = 32
load_model = True
num_iter =200
show_every = 100
shuffle=True
shuffle_every=10
save_every=1000
max_num_img=300
batch_size=300#must be smaller or equal than max_num_img
LR_enc=LR_gen= 0.001
LR_disc=0.001
l_reg=100
latent_dim=2

mirror=True
rotate=True

input_corruption='noise'#can be one of: 'None','noise' or 'holes'
latent_distribution='gauss'#can be one of: 'uniform' or 'gauss'

nef=20
ngf=20

#----- specifiy the figure
random.seed(1)
torch.manual_seed(1)

img_path='/Users/Giaco/Documents/Elektrotechnik-Master/image_processing/my_deep_image_prior/data/rose/'
mask_path='/Users/Giaco/Documents/Elektrotechnik-Master/image_processing/my_deep_image_prior/data/inpainting/library_mask.png'

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
print('size of img_torch_array: '+str(img_torch_array.size()))

#---load mask
if input_corruption=='None':
  net_input=img_torch_array.detach().clone()
elif input_corruption=='holes':
  masked_torch_array=torch.zeros(n_img_loaded,3,imsize,imsize)
  for j in range(n_img_loaded):
    masked_torch_array[j,:,:,:]=torch.from_numpy(pil_to_np(get_mask(imsize,N=10,S=10))).type(dtype)
  masked_images=img_torch_array*masked_torch_array
  neg_masked_images=img_torch_array.detach().clone().masked_fill_((masked_images*-1+1).type(torch.ByteTensor), -1)
  net_input=neg_masked_images.detach().clone()
elif input_corruption=='noise':
  net_input=img_torch_array+torch.randn(n_img_loaded, 3, imsize, imsize)*0.1
else:
  raise ValueError('the value: "input_corruption" must be one of {None,noise,holes}!')

#----torch inizializations
print('inputshape '+str(net_input.size()))

#---specify optimizer
OPTIMIZER = 'adam'

if not os.path.exists('saved_models/'):
    os.mkdir('saved_models')
if not os.path.exists('aae_images/') and PLOT:
    os.mkdir('aae_images')
#-------image specific settigns----------
encoder = dc_encoder(latent_dim,imsize,ndf=nef)
generator = dc_generator(latent_dim,ngf=ngf,nc=3,imgsize=imsize)
discriminator = dfc_discriminator(latent_dim)

optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=LR_enc)
optimizer_gen= torch.optim.Adam(generator.parameters(), lr=LR_gen)
optimizer_disc=torch.optim.Adam(discriminator.parameters(), lr=LR_disc)
state_epoch=0
if load_model:
  print('reload model....')
  state_dict_enc=torch.load('saved_models/aae_enc_'+str(imsize)+'.pkl')
  state_dict_gen=torch.load('saved_models/aae_gen_'+str(imsize)+'.pkl')
  state_dict_disc=torch.load('saved_models/aae_disc_'+str(imsize)+'.pkl')
  state_epoch=state_dict_gen['epoch']
  encoder.load_state_dict(state_dict_enc['model_state'])
  generator.load_state_dict(state_dict_gen['model_state'])
  discriminator.load_state_dict(state_dict_disc['model_state'])
  optimizer_enc.load_state_dict(state_dict_enc['optimizer_state'])
  optimizer_gen.load_state_dict(state_dict_gen['optimizer_state'])
  optimizer_disc.load_state_dict(state_dict_disc['optimizer_state'])

np_enc = sum(np.prod(list(p.size())) for p in encoder.parameters())
np_gen  = sum(np.prod(list(p.size())) for p in generator.parameters())
np_tot=np_enc+np_gen
print ('Number of params in dc_autoencoder: %d' % np_tot)

def disc_loss_fuction(d,d_hat,batchSize):
  # print('detect reals with prob: '+str(torch.sum(d)/batchSize))
  # print('detect fakes with prob: '+str(torch.sum(1-d_hat)/batchSize))
  loss=torch.sum(-torch.log(d)-torch.log(1-d_hat))
  return loss/batchSize

def ae_loss_function(inpainted,orig,dhat,batchSize):
  de=inpainted-orig 
  loss=torch.sum(torch.mul(de,de))
  reg_loss=l_reg*torch.sum(-torch.log(dhat))
  # print('regularization loss: '+str(reg_loss/batchSize))
  # print('reconstruction loss: '+str(loss/batchSize))
  loss+=reg_loss
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
      if input_corruption=='holes':
        net_input=img_torch_array.detach().clone().masked_fill_((-img_torch_array*masked_torch_array[shuffle_idx]+1).type(torch.ByteTensor), -1)[shuffle_idx]
      elif input_corruption=='noise':
        print('new noise:')
        net_input=net_input[shuffle_idx]+torch.randn(n_img_loaded, 3, imsize, imsize)*0.1
      else:
        net_input=net_input[shuffle_idx]
      img_torch_array=img_torch_array[shuffle_idx]

    for idx in range(len(batch_idx)-1):
      batchSize=batch_idx[idx+1]-batch_idx[idx]
      optimizer_gen.zero_grad()
      optimizer_enc.zero_grad()
      optimizer_disc.zero_grad()

#-----propagate the autoencoder
      latent_out=encoder(net_input[batch_idx[idx]:batch_idx[idx+1],:]) 
      out = generator(latent_out)

#----discriminator training
      if latent_distribution=='uniform':
        latent_noise=torch.rand(batchSize, latent_dim)-0.5
      else:
        latent_noise=torch.randn(batchSize, latent_dim)
      d=discriminator(latent_noise)
      d_hat1 = discriminator(torch.squeeze(latent_out).detach())
      loss_disc = disc_loss_fuction(d,d_hat1,batchSize)
      loss_disc.backward()
      optimizer_disc.step()

#-----generator training
      d_hat2 = discriminator(torch.squeeze(latent_out))
      ae_loss=ae_loss_function(out,img_torch_array[batch_idx[idx]:batch_idx[idx+1],:],d_hat2,batchSize)
      ae_loss.backward()
      optimizer_enc.step()
      optimizer_gen.step()
      train_loss+=ae_loss.item()*batchSize

    epoch_loss=train_loss
    print('Iteration: '+str(i)+'   '+'Loss: '+str(epoch_loss/imsize**2))
    if  PLOT and (i+1) % show_every == 0:
        print('save plot ...')
        for n_im in range(np.minimum(10,batch_size)):
          out_np = out[n_im,:].transpose(0,1).transpose(1,2).detach().numpy()
          corrupted_np = torch.clamp(net_input[+n_im,:].transpose(0,1).transpose(1,2).detach(),0,1).numpy()
          orig_np=img_torch_array[n_im,:].transpose(0,1).transpose(1,2).detach().numpy()
          save_comparison_plot(corrupted_np,out_np,orig_np,'aae_images/'+str(i)+'_'+str(n_im))
    if latent_dim==2:
      np_latent_out=torch.squeeze(latent_out).detach().numpy()
      np_latent_noise=latent_noise.detach().numpy()
      plt.clf()
      plt.plot(np_latent_out[:,0], np_latent_out[:,1], 'ro',np_latent_noise[:,0],np_latent_noise[:,1],'b*')
      plt.savefig('aae_images/latent_space_'+str(i))

    if (i+1)%save_every==0:
      print('save model ...')
      torch.save({'epoch': i, 'model_state': encoder.state_dict(),'optimizer_state': optimizer_enc.state_dict()}, 'saved_models/aae_enc_'+str(imsize)+'.pkl')
      torch.save({'epoch': i, 'model_state': generator.state_dict(),'optimizer_state': optimizer_gen.state_dict()}, 'saved_models/aae_gen_'+str(imsize)+'.pkl')
      torch.save({'epoch': i, 'model_state': discriminator.state_dict(),'optimizer_state': optimizer_disc.state_dict()}, 'saved_models/aae_disc_'+str(imsize)+'.pkl')      
    i += 1

    return epoch_loss

#-----call optimizer and save stuff ----
if TRAINING:
  print('start training ...')
  for j in range(num_iter):
    closure()
  torch.save({'epoch': i, 'model_state': generator.state_dict(),'optimizer_state': optimizer_gen.state_dict()}, 'saved_models/aae_gen_'+str(imsize)+'.pkl')
  torch.save({'epoch': i, 'model_state': encoder.state_dict(),'optimizer_state': optimizer_enc.state_dict()}, 'saved_models/aae_enc_'+str(imsize)+'.pkl')
  torch.save({'epoch': i, 'model_state': discriminator.state_dict(),'optimizer_state': optimizer_disc.state_dict()}, 'saved_models/aae_disc_'+str(imsize)+'.pkl')   




