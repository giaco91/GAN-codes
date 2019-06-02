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
import torchvision.utils as vutils

from common_utils import *

#---global settings

dtype = torch.FloatTensor
PLOT = True
TRAINING = True
imsize = 32
load_model = True
num_iter =500
show_every = 25
shuffle=True
shuffle_every=10
save_every=25
max_num_img=800
batch_size=80#must be smaller or equal than max_num_img
LR_gen= 0.0001
LR_disc=0.0001
nz=5

mirror=True
rotate=True

ngf=30
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
#---specify optimizer
OPTIMIZER = 'adam'

if not os.path.exists('saved_models/'):
    os.mkdir('saved_models')
if not os.path.exists('dc_gan_images/') and PLOT:
    os.mkdir('dc_gan_images/')
#-------image specific settigns----------
generator = dc_generator(nz,ngf=ngf,nc=3,imgsize=imsize)
discriminator = dc_discriminator(imsize,ndf=ndf,nc=3,max_depth=7)
optimizer_gen= torch.optim.Adam(generator.parameters(), lr=LR_gen)
optimizer_disc= torch.optim.Adam(discriminator.parameters(), lr=LR_disc)
state_epoch=0
if load_model:
  print('reload model....')
  state_dict_gen=torch.load('saved_models/dc_gan_generator'+str(imsize)+'.pkl')
  state_dict_disc=torch.load('saved_models/dc_gan_discriminator'+str(imsize)+'.pkl')
  state_epoch=state_dict_gen['epoch']
  generator.load_state_dict(state_dict_gen['model_state'])
  discriminator.load_state_dict(state_dict_disc['model_state'])
  optimizer_gen.load_state_dict(state_dict_gen['optimizer_state'])
  optimizer_disc.load_state_dict(state_dict_disc['optimizer_state'])


# Compute number of parameters
np_gen  = sum(np.prod(list(p.size())) for p in generator.parameters())
print ('Number of params in dc_generator: %d' % np_gen)

def disc_loss_fuction(d,d_hat,batchSize):
  # print('detect reals with prob: '+str(torch.sum(d)/batchSize))
  # print('detect fakes with prob: '+str(torch.sum(1-d_hat)/batchSize))
  loss=torch.sum(-torch.log(d)-torch.log(1-d_hat))
  return loss/batchSize

def gen_loss_function(d_hat,batchSize):
    loss=torch.sum(-torch.log(d_hat))
    return loss/batchSize
criterion = nn.BCELoss()
fixed_noise = torch.randn(batch_size, nz, 1, 1)
real_label = 1
fake_label = 0

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
      shuffle_idx=torch.randperm(img_torch_array.size()[0])
      img_torch_array=img_torch_array[shuffle_idx]

    for idx in range(len(batch_idx)-1):
      batchSize=batch_idx[idx+1]-batch_idx[idx]
      optimizer_gen.zero_grad()
      optimizer_disc.zero_grad()

#-----generation
      noise = torch.randn(batchSize, nz, 1, 1)
      out = generator(noise)
 
#----discriminator training
      d=discriminator(img_torch_array[batch_idx[idx]:batch_idx[idx+1],:])
      d_hat1 = discriminator(out.detach())
      loss_disc = disc_loss_fuction(d,d_hat1,batchSize)
      loss_disc.backward()
      # label = torch.full((batchSize,), real_label)
      # loss_disc_real=criterion(d,label)
      # loss_disc_real.backward()
      # label.fill_(fake_label)
      # loss_disc_fake=criterion(d_hat1,label)
      # loss_disc_fake.backward()
      # loss_disc=loss_disc_real.item()+loss_disc_fake.item()
      loss_disc.item()
      D = d.mean().item()
      D_hat1=d_hat1.mean().item()
      optimizer_disc.step()

#---- generator training
      d_hat2=discriminator(out)
      # label.fill_(real_label)
      # loss_gen=criterion(d_hat2,label)
      loss_gen=gen_loss_function(d_hat2,batchSize)
      loss_gen.backward()
      D_hat2=d_hat2.mean().item()
      optimizer_gen.step()

    # if  PLOT and (i+1) % show_every == 0:
    #     # n_im=i%n_img_loaded
    #     print('save plot ...')
    #     for n_im in range(np.minimum(10,batch_size)):
    #       out_np = torch.clamp(out[n_im,:].transpose(0,1).transpose(1,2).detach(),0,1).numpy()
    #       masked_np = torch.clamp(net_input[+n_im,:].transpose(0,1).transpose(1,2).detach(),0,1).numpy()
    #       orig_np=img_torch_array[n_im,:].transpose(0,1).transpose(1,2).detach().numpy()
    #       save_comparison_plot(masked_np,out_np,orig_np,'conv_ae_images_latent_disc/'+str(i)+'_'+str(n_im))

      print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
            % (i, num_iter+state_epoch, idx, len(batch_idx)-1,
               loss_disc, loss_gen.item(), D, D_hat1, D_hat2))
      if (i+1) % show_every == 0:
          vutils.save_image(img_torch_array[batch_idx[idx]:batch_idx[idx+1],:],
                  '%s/real_samples.png' % 'dc_gan_images',
                  normalize=True)
          fake = generator(fixed_noise)
          vutils.save_image(fake.detach(),
                  '%s/fake_samples_epoch_%03d.png' % ('dc_gan_images', i),
                  normalize=True)

    if (i+1)%save_every==0:
      print('save model ...')
      torch.save({'epoch': i, 'model_state': generator.state_dict(),'optimizer_state': optimizer_gen.state_dict()}, 'saved_models/dc_gan_generator'+str(imsize)+'.pkl')  
      torch.save({'epoch': i, 'model_state': discriminator.state_dict(),'optimizer_state': optimizer_disc.state_dict()}, 'saved_models/dc_gan_discriminator'+str(imsize)+'.pkl')    
    i += 1

    return

#-----call optimizer and save stuff ----
if TRAINING:
  print('start training ...')
  for j in range(num_iter):
    closure()
  torch.save({'epoch': i, 'model_state': generator.state_dict(),'optimizer_state': optimizer_gen.state_dict()}, 'saved_models/dc_gan_generator'+str(imsize)+'.pkl')
  torch.save({'epoch': i, 'model_state': discriminator.state_dict(),'optimizer_state': optimizer_disc.state_dict()}, 'saved_models/dc_gan_discriminator'+str(imsize)+'.pkl') 
#---- testing 

# corrupted_img=net_input+(-2+j*0.1)
# if PLOT:
#   N=20
#   corrupted_img=torch.randn(N, latent_dim, 1, 1, device='cpu')
#   with torch.no_grad():
#     latent_out=encoder(net_input[0:100,:])
#     latent_std=torch.sqrt(torch.sum(torch.mul(latent_out,latent_out))/100)
#     print('standard deviation in latent_space: '+str(latent_std))
#     corrupted_img=torch.randn(N, latent_dim, 1, 1, device='cpu')*latent_std
#     inpainted_img=torch.clamp(generator(corrupted_img), 0, 1).transpose(1,2).transpose(2,3).detach().numpy()
#   original_img = img_torch_array[0:1,:].transpose(1,2).transpose(2,3).detach().numpy()[0,:]
#   corrupted_img = masked_images[0:1,:].transpose(1,2).transpose(2,3).detach().numpy()[0,:]
#   for j in range(N):
#     inpainted_img_j = inpainted_img[j:j+1,:]
#     save_comparison_plot(corrupted_img[0,:],inpainted_img_j[0,:],original_img[0,:],'conv_ae_images_latent_disc/'+str(j))



