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
import copy

from common_utils import *

#---global settings

dtype = torch.FloatTensor
TRAINING = True
imsize = 32
load_model = False
num_iter =20
selection_every= 10#is also show every
shuffle=True
shuffle_every=10
save_every=20
max_num_img=80
batch_size=80#must be smaller or equal than max_num_img
LR_gen= 0.0005
LR_disc=0.0005
nz=5
n_generator=9
n_survivor=3 #must be a divisor of n_generator
mirror=True
rotate=True
influence_factor=1#in [0,1] the amount of how much the new fake generate influences the history
buffer_size=1400
buffer_size=max(buffer_size,batch_size*n_generator)#buffersize must at least contain all outputs of the generators
ngf=20
ndf=20

if n_generator%n_survivor!=0:
  raise ValueError('n_survivor must be a divisor of n_generator')
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
if not os.path.exists('e_gan_images/'):
    os.mkdir('e_gan_images/')
#-------image specific settigns----------
def change_lr(optimizer,lr):
  for param_group in optimizer.param_groups:
        param_group['lr'] = lr

id_list=[]
survivor_list=[]
optimizer_survivor_list=[]
lr_noise=(torch.rand(n_survivor)-0.5)*LR_gen
loss_codes=torch.rand(n_survivor,3)-0.5
loss_code_list=[]
lr_list=[]
for g in range(n_survivor):
 id_list.append([g+1])
 survivor_list.append(dc_generator(nz,ngf=ngf,nc=3,imgsize=imsize))
 lr_g=LR_gen+lr_noise[g]
 lr_list.append(lr_g)
 optimizer_survivor_list.append(torch.optim.Adam(survivor_list[g].parameters(), lr=lr_g))
 loss_code_list.append(loss_codes[g,:])
discriminator = dc_discriminator(imsize,ndf=ndf,nc=3,max_depth=7)
optimizer_disc= torch.optim.Adam(discriminator.parameters(), lr=LR_disc)
state_epoch=0
if load_model:
  print('reload model....')
  for g in range(n_survivor):
    state_dict_gen=torch.load('saved_models/e_gan_generator'+str(imsize)+'_'+str(g)+'.pkl')
    survivor_list[g].load_state_dict(state_dict_gen['model_state'])
    optimizer_survivor_list[g].load_state_dict(state_dict_gen['optimizer_state'])
    id_list[g]=state_dict_gen['id']
    lr_g=state_dict_gen['lr']
    change_lr(optimizer_survivor_list[g],lr_g)
  state_dict_disc=torch.load('saved_models/e_gan_discriminator'+str(imsize)+'.pkl')
  state_epoch=state_dict_disc['epoch']
  discriminator.load_state_dict(state_dict_disc['model_state'])
  optimizer_disc.load_state_dict(state_dict_disc['optimizer_state'])

def softmax(torch_array):
  torch_exp_array=torch.exp(torch_array)
  return torch_exp_array/torch.sum(torch_exp_array)

def get_relative_distance(id_1,id_2):
  if len(id_1)!=len(id_2):
    raise ValueError('the two ids must be equally long!')
  for i in range(len(id_1)):
    idx=i+1
    if id_1[:idx]!=id_2[:idx]:
      return len(id_2)-(i)
  return 0


def mutation(survivor_list,optimizer_survivor_list,lr_list,loss_code_list,id_list):
  new_generator_list=[]
  new_optimizer_list=[]
  new_lr_list=[]
  new_loss_code_list=[]
  new_loss_code_softmax_list=[]
  new_id_list=[]
  n_mutation=int(n_generator/n_survivor)
  (np.random.rand(n_survivor)-0.5)*LR_gen
  for s in range(n_survivor):
    new_id_list.append(id_list[s]+[0])
    new_generator_list.append(survivor_list[s])
    new_optimizer_list.append(optimizer_survivor_list[s])
    new_lr_list.append(lr_list[s])
    new_loss_code_list.append(loss_code_list[s])
    new_loss_code_softmax_list.append(softmax(loss_code_list[s]))
    for m in range(n_mutation-1):
      new_id_list.append(id_list[s]+[m+1])
      new_generator=copy.deepcopy(survivor_list[s])
      new_generator_list.append(new_generator)
      new_lr=lr_list[s]+(np.random.rand()-0.5)*lr_list[s]#noise mutation
      new_lr_list.append(new_lr)
      new_optimizer=torch.optim.Adam(new_generator.parameters(), lr=new_lr)
      new_optimizer.load_state_dict(optimizer_survivor_list[s].state_dict())
      change_lr(new_optimizer,new_lr)
      new_optimizer_list.append(new_optimizer)
      if np.random.rand()>0.9:
        new_loss_code=loss_code_list[s][torch.randperm(3)]
        new_loss_code_list.append(new_loss_code)#permutation mutation
      else:
        new_loss_code=loss_code_list[s]+(torch.rand(3)-0.5)*0.1
        new_loss_code_list.append(new_loss_code)#noise mutation
      new_loss_code_softmax_list.append(softmax(new_loss_code))

  return new_generator_list,new_optimizer_list,new_lr_list,new_loss_code_list,new_loss_code_softmax_list,new_id_list
 

def gen_loss_function(d_hat,loss_code_softmax,batchSize):
    loss=loss_code_softmax[0]*torch.sum(-torch.log(d_hat))
    delta=1-d_hat
    loss+=loss_code_softmax[1]*torch.sum(torch.log(delta))
    loss+=loss_code_softmax[2]*torch.sum(torch.mul(delta,delta))
    return loss/batchSize

criterion = nn.BCELoss()
fixed_noise = torch.randn(20, nz, 1, 1)
fake_label_buffer = torch.full((buffer_size,), 0)
fake_label = 0

#----- training loop--------
batch_idx=[0]
for b in range(int(np.floor(n_img_loaded/batch_size))):
  batch_idx.append((b+1)*batch_size)
if n_img_loaded>batch_size*(len(batch_idx)-1):
  batch_idx.append(n_img_loaded)

new_generator_list,new_optimizer_list,new_lr_list,new_loss_code_list,new_loss_code_softmax_list,new_id_list=mutation(survivor_list,optimizer_survivor_list,lr_list,loss_code_list,id_list)

#----fill buffer----
print('initializing buffer')
buffer_size=max(n_generator*batch_size,buffer_size)
buffer_fakes=torch.zeros(buffer_size,3,imsize,imsize)
n_noise=int(buffer_size/n_generator)
noise = torch.randn(n_noise, nz, 1, 1)
rest=buffer_size-n_generator*n_noise
for g in range(n_generator):
  buffer_fakes[g*n_noise:(g+1)*n_noise,:,:,:]=new_generator_list[g](noise).detach()
if rest>0:
  buffer_fakes[n_generator:n_generator+rest,:,:,:]=new_generator_list[0](torch.randn(rest, nz, 1, 1)).detach()

i = state_epoch
def closure():  
    global i,net_input,img_torch_array,new_generator_list,new_optimizer_list,new_lr_list,new_loss_code_list,new_loss_code_softmax_list,new_id_list,buffer_fakes
    d_hat_nparray=np.zeros(n_generator)
#---shuffle the data
    if shuffle and (i+1)%shuffle_every==0:
      shuffle_idx=torch.randperm(img_torch_array.size()[0])
      img_torch_array=img_torch_array[shuffle_idx]
    

#---batch loop
    for idx in range(len(batch_idx)-1):
      buffer_fakes=buffer_fakes[torch.randperm(buffer_size)]
      batchSize=batch_idx[idx+1]-batch_idx[idx]
      optimizer_disc.zero_grad() 
      n_update=int(influence_factor*batchSize)
#-----generation
      noise = torch.randn(batchSize, nz, 1, 1)
      out_list=[]
      for g in range(n_generator):
        fake_images_g=new_generator_list[g](noise)
        out_list.append(fake_images_g)
        buffer_fakes[g*n_update:(g+1)*n_update,:]=fake_images_g[:n_update,:].detach()
 
#----discriminator training
      d=discriminator(img_torch_array[batch_idx[idx]:batch_idx[idx+1],:])
      label = torch.full((batchSize,), 1)
      loss_disc_real=criterion(d,label)
      loss_disc_real.backward()
      loss_disc_fake=0
      # for g in range(n_generator):
      #   d_hat1=discriminator(out_list[g].detach())
      #   loss_disc_fake+=criterion(d_hat1,label)
      #   D_hat1+=d_hat1.mean().item()
      d_hat1=discriminator(buffer_fakes)
      loss_disc_fake=criterion(d_hat1,fake_label_buffer)
      loss_disc_fake*=(batchSize/buffer_size)
      loss_disc_fake.backward()
      D = d.mean().item()
      optimizer_disc.step()
      #D_hat1=d_hat1.mean().item()*(batchSize/buffer_size)

#---- generator training
      D_hat=0
      for g in range(n_generator):
        new_optimizer_list[g].zero_grad()
        d_hat=discriminator(out_list[g])
        loss_gen=gen_loss_function(d_hat,new_loss_code_softmax_list[g],batchSize)
        loss_gen.backward()
        new_optimizer_list[g].step()
        d_hat_nparray[g]+=d_hat.mean().item()
        D_hat+=d_hat.mean().item()

      D_hat/=n_generator

        

      print('[%d/%d][%d/%d] D(x): %.4f D(G(z)): %.4f'
            % (i+1, num_iter+state_epoch, idx+1, len(batch_idx)-1, D, D_hat))
    if (i+1) % selection_every == 0:
      #-----selection
        print('selection and diversification....')
        d_hat_sorted=d_hat_nparray.argsort()
        print('sorted d_hat:'+str(np.sort(d_hat_nparray)))
        loss_code_list_softmax=[]
        for s in range(n_survivor):
          idx_s=d_hat_sorted[-(s+1)]
          survivor_list[s]=new_generator_list[idx_s]
          optimizer_survivor_list[s]=new_optimizer_list[idx_s]
          lr_list[s]=new_lr_list[idx_s]
          loss_code_list[s]=new_loss_code_list[idx_s]
          loss_code_list_softmax.append(softmax(loss_code_list[s]))
          id_list[s]=new_id_list[idx_s]
        print('survivors learning rates: '+str(lr_list))
        print('survivors loss codes: '+str(loss_code_list_softmax))
        # print('survivors id: '+str(id_list))
        counter=0
        s_rd=0
        for k in range(len(id_list)-1):
          for l in range(k+1,len(id_list)):
            counter+=1
            s_rd+=get_relative_distance(id_list[k],id_list[l])
        m_rd=s_rd/counter
        print('mean relative distance: %.3f' % (m_rd))
        #mutation and diversivication
        new_generator_list,new_optimizer_list,new_lr_list,new_loss_code_list,new_loss_code_softmax_list,new_id_list=mutation(survivor_list,optimizer_survivor_list,lr_list,loss_code_list,id_list)
        print('new generation')
        vutils.save_image(img_torch_array[batch_idx[idx]:batch_idx[idx+1],:],
                '%s/real_samples.png' % 'e_gan_images',
                normalize=True)


        for s in range(n_survivor):
          fake = survivor_list[s](fixed_noise)
          vutils.save_image(fake.detach(),
                  '%s/fake_samples_epoch_%03d_%01d.png' % ('e_gan_images', i,s),
                  normalize=True)

    if (i+1)%save_every==0:
      print('save model ...')
      for s in range(n_survivor):
        torch.save({'epoch': i, 'model_state': survivor_list[s].state_dict(),'loss_code': loss_code_list[s], 'lr':lr_list[s],'id': id_list[s],'optimizer_state': optimizer_survivor_list[s].state_dict()}, 'saved_models/e_gan_generator'+str(imsize)+'_'+str(s)+'.pkl')  
      torch.save({'epoch': i, 'model_state': discriminator.state_dict(),'optimizer_state': optimizer_disc.state_dict()}, 'saved_models/e_gan_discriminator'+str(imsize)+'.pkl')    
    i += 1

    return

#-----call optimizer and save stuff ----
if TRAINING:
  print('start training ...')
  for j in range(num_iter):
    closure()
  for s in range(n_survivor):
    torch.save({'epoch': i, 'model_state': survivor_list[s].state_dict(),'loss_code': loss_code_list[s], 'lr':lr_list[s],'id': id_list[s],'optimizer_state': optimizer_survivor_list[s].state_dict()}, 'saved_models/e_gan_generator'+str(imsize)+'_'+str(s)+'.pkl')  
  torch.save({'epoch': i, 'model_state': discriminator.state_dict(),'optimizer_state': optimizer_disc.state_dict()}, 'saved_models/e_gan_discriminator'+str(imsize)+'.pkl') 
#---- testing 





