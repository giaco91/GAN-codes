import matplotlib.pyplot as plt
import glob
import time
import os
import random
import numpy as np
from models.dc_generator import dc_generator
from models.dc_discriminator import dc_discriminator
from models.dc_mult_discriminator import dc_mult_discriminator
import torch
import torch.optim
import time
import torchvision.utils as vutils

from common_utils import *

#---global settings

dtype = torch.FloatTensor
PLOT = True
TRAINING = True
imsize = 16
load_model = False
num_iter =200
show_every = 10
shuffle=True
shuffle_every=10
save_every=25
max_num_img=160
batch_size=80#must be smaller or equal than max_num_img
LR_gen= 0.001
LR_disc=0.001
n_generators=5
n_discriminators=2
nz=5

mirror=True
rotate=True

ngf=50
ndf=10

#----- specifiy the figure
random.seed(1)
torch.manual_seed(1)

# a low res rose image
img_path='/Users/Giaco/Documents/Elektrotechnik-Master/image_processing/my_deep_image_prior/data/rose/'
# mask_path='/Users/Giaco/Documents/Elektrotechnik-Master/image_processing/my_deep_image_prior/data/inpainting/library_mask.png'

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

#----torch inizializations
if not os.path.exists('saved_models/'):
    os.mkdir('saved_models')
if not os.path.exists('m_gan_images/') and PLOT:
    os.mkdir('m_gan_images/')
#-------image specific settigns----------
generator_list=[]
optimizer_gen_list=[]
discriminator_list=[]
optimizer_disc_list=[]
for g in range(n_generators):
 generator_list.append(dc_generator(nz,ngf=ngf,nc=3,imgsize=imsize))
 optimizer_gen_list.append(torch.optim.Adam(generator_list[g].parameters(), lr=LR_gen))
for d in range(n_discriminators):
  discriminator_list.append(dc_discriminator(imsize,ndf=ndf,nc=3,max_depth=7))
  optimizer_disc_list.append(torch.optim.Adam(discriminator_list[d].parameters(), lr=LR_disc))

state_epoch=0
if load_model:
  print('reload model....')
  for g in range(n_generators):
    state_dict_gen=torch.load('saved_models/m_gan_generator'+str(imsize)+'_'+str(g)+'.pkl')
    generator_list[g].load_state_dict(state_dict_gen['model_state'])
    optimizer_gen_list[g].load_state_dict(state_dict_gen['optimizer_state'])
  for d in range(n_discriminators):
    state_dict_disc=torch.load('saved_models/m_gan_discriminator'+str(imsize)+'.pkl')
    discriminator_list[d].load_state_dict(state_dict_disc['model_state'])
    optimizer_disc_list[d].load_state_dict(state_dict_disc['optimizer_state'])
  state_epoch=state_dict_gen['epoch']

criterion = nn.BCELoss()
fixed_noise = torch.randn(50, nz, 1, 1)

#----- training loop--------
batch_idx=[0]
for b in range(int(np.floor(n_img_loaded/batch_size))):
  batch_idx.append((b+1)*batch_size)
if n_img_loaded>batch_size*(len(batch_idx)-1):
  batch_idx.append(n_img_loaded)


i = state_epoch
def closure():  
    global i,net_input,img_torch_array

    train_loss=0
    if shuffle and (i+1)%shuffle_every==0:
      shuffle_idx=torch.randperm(img_torch_array.size()[0])
      img_torch_array=img_torch_array[shuffle_idx]

    for idx in range(len(batch_idx)-1):
      batchSize=batch_idx[idx+1]-batch_idx[idx]

#-----generation
      real_label = torch.full((batchSize,), 1)
      fake_label = torch.full((batchSize,), 0)
      noise = torch.randn(batchSize, nz, 1, 1)
      out_list=[]
      for g in range(n_generators):
        out_list.append(generator_list[g](noise))
 
#----discriminator between real and fake images training
      D=0
      for d in range(n_discriminators):
        optimizer_disc_list[d].zero_grad()
        d=discriminator_list[d](img_torch_array[batch_idx[idx]:batch_idx[idx+1],:])
        loss_disc_real=criterion(d,real_label)
        loss_disc_real.backward()
        D += d.mean().item()
      D/=n_discriminators

      D_hat1=0
      for d in range(n_discriminators):
        loss_disc_fake=0
        for g in range(n_generators):
          d_hat1=discriminator_list[d](out_list[g].detach())
          loss_disc_fake+=criterion(d_hat1,fake_label)
          D_hat1+=d_hat1.mean().item()
        loss_disc_fake/=n_generators
        loss_disc_fake.backward()
        optimizer_disc_list[d].step()
      D_hat1/=n_generators*n_discriminators

#---- generator training
      minibatch_loss_gen=np.zeros(n_generators)
      for g in range(n_generators):
        loss_gen=0
        for d in range(n_discriminators):
          optimizer_gen_list[g].zero_grad()
          d_hat=discriminator_list[d](out_list[g])
          loss_gen+=criterion(d_hat,real_label)
        loss_gen/=n_discriminators
        loss_gen.backward()
        optimizer_gen_list[g].step()
        minibatch_loss_gen[g]+=loss_gen.item()

      print('[%d/%d][%d/%d] D(x): %.4f D(G(z)): %.4f '
            % (i+1, num_iter+state_epoch, idx+1, len(batch_idx)-1, D, D_hat1))

    if (i+1) % show_every == 0:
        vutils.save_image(img_torch_array[batch_idx[idx]:batch_idx[idx+1],:],
                '%s/real_samples.png' % 'm_gan_images',
                normalize=True)
        for g in range(n_generators):
          fake = generator_list[g](fixed_noise)
          vutils.save_image(fake.detach(),
                  '%s/fakes_epoch_%03d_generator_%03d.png' % ('m_gan_images', i+1,g),
                  normalize=True)

    if (i+1)%save_every==0:
      print('save model ...')
      for g in range(n_generators):
        torch.save({'epoch': i, 'model_state': generator_list[g].state_dict(), 'optimizer_state': optimizer_gen_list[g].state_dict()}, 'saved_models/m_gan_generator'+str(imsize)+'_'+str(g)+'.pkl')
      for d in range(n_discriminators):
        torch.save({'epoch': i, 'model_state': discriminator_list[d].state_dict(),'optimizer_state': optimizer_disc_list[d].state_dict()}, 'saved_models/m_gan_discriminator'+str(imsize)+'_'+str(d)+'.pkl')    
 
    i += 1

    return

#-----call optimizer and save stuff ----
if TRAINING:
  print('start training ...')
  for j in range(num_iter):
    closure()

  for g in range(n_generators):
    torch.save({'epoch': i, 'model_state': generator_list[g].state_dict(), 'optimizer_state': optimizer_gen_list[g].state_dict()}, 'saved_models/m_gan_generator'+str(imsize)+'_'+str(g)+'.pkl') 
  for d in range(dc_discriminators):
    torch.save({'epoch': i, 'model_state': discriminator_list[d].state_dict(),'optimizer_state': optimizer_disc_list[d].state_dict()}, 'saved_models/m_gan_discriminator'+str(imsize)+'_'+str(d)+'.pkl')




