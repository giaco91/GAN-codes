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
from models.dc_discriminator import dc_discriminator
import torch
import torch.optim
import time
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms

from common_utils import *

#---global settings
dtype = torch.FloatTensor
PLOT = True
TRAINING = True
imsize = 32
load_model = True
num_iter =1
show_every = 1
show_every_n_batch = 1
shuffle=True
shuffle_every=1
save_every=10
max_num_img=20
batch_size=50#must be smaller or equal than max_num_img

LR_enc= 0.00
LR_gen= 0.00
LR_disc_latent=0.00
LR_disc_image=0.00
l_reg_latent=1
l_reg_image=1
l_reg_recon=1
latent_dim=100

input_corruption='None'#can be one of: 'None','noise' or 'holes'
latent_distribution='gauss'#can be one of: 'uniform' or 'gauss'
mirror=True
rotate=True

nef=60
ngf=60
ndf=60

num_workers=2
#----- specifiy the figure
random.seed(1)
torch.manual_seed(1)
#img_path='/Users/Giaco/Documents/Elektrotechnik-Master/image_processing/my_deep_image_prior/data/flower_images/'
img_path='/Users/Giaco/Documents/Elektrotechnik-Master/image_processing/my_deep_image_prior/data/rose/'
mask_path='/Users/Giaco/Documents/Elektrotechnik-Master/image_processing/my_deep_image_prior/data/inpainting/library_mask.png'

# #load images
# img_list=[]
# i=0
# for img_name in os.listdir(img_path):
#   if (img_name.endswith(".jpg") or img_name.endswith(".png")) and i<max_num_img:
#     img_pil=square_crop(img_path+img_name)
#     img_pil=resize_to_height_ref(img_pil,imsize)
#     img_list.append(pil_to_np(img_pil))
#     i+=1
#     #----data augmentation-----
#     if mirror and i<max_num_img:
#       img_pil_mirror=PIL.ImageOps.mirror(img_pil)
#       img_list.append(pil_to_np(img_pil_mirror))
#       i+=1
#       if rotate and i<max_num_img:
#         img_list.append(pil_to_np(img_pil_mirror.rotate(90, expand=False)))
#         i+=1
#         if i<max_num_img:
#           img_list.append(pil_to_np(img_pil_mirror.rotate(180, expand=False)))
#           i+=1
#         if i<max_num_img:
#           img_list.append(pil_to_np(img_pil_mirror.rotate(270, expand=False)))
#           i+=1
#     if rotate and i<max_num_img:
#       img_list.append(pil_to_np(img_pil.rotate(90, expand=False)))
#       i+=1
#       if i<max_num_img:
#         img_list.append(pil_to_np(img_pil.rotate(180, expand=False)))
#         i+=1
#       if i<max_num_img:
#         img_list.append(pil_to_np(img_pil.rotate(270, expand=False)))
#         i+=1

# n_img_loaded=len(img_list)
# print('number of loaded images: '+str(n_img_loaded))
# img_shape=img_list[0].shape
# img_np_array=np.zeros([len(img_list),img_shape[0],img_shape[1],img_shape[2]])
# for j in range(len(img_list)):
#   img_np_array[j,:]=img_list[j]
# img_torch_array=torch.from_numpy(img_np_array).type(dtype)
# print('size of img_torch_array: '+str(img_torch_array.size()))

root_to_dataset='/Users/Giaco/Documents/Elektrotechnik-Master/image_processing/my_deep_image_prior/data/'
trainset=dset.CIFAR10(root=root_to_dataset, train=True, transform=transforms.Compose([
                                   transforms.Resize(imsize),
                                   transforms.CenterCrop(imsize),
                                   transforms.ToTensor(),
                               ]), download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)


#---load mask
# if input_corruption=='None':
#   net_input=img_torch_array.detach().clone()
# elif input_corruption=='holes':
#   masked_torch_array=torch.zeros(n_img_loaded,3,imsize,imsize)
#   for j in range(n_img_loaded):
#     masked_torch_array[j,:,:,:]=torch.from_numpy(pil_to_np(get_mask(imsize,N=10,S=10))).type(dtype)
#   masked_images=img_torch_array*masked_torch_array
#   neg_masked_images=img_torch_array.detach().clone().masked_fill_((masked_images*-1+1).type(torch.ByteTensor), -1)
#   net_input=neg_masked_images.detach().clone()
# elif input_corruption=='noise':
#   net_input=img_torch_array+torch.randn(n_img_loaded, 3, imsize, imsize)*0.1
# else:
#   raise ValueError('the value: "input_corruption" must be one of {None,noise,holes}!')

#----torch inizializations
# print('inputshape '+str(net_input.size()))

#---specify optimizer

if not os.path.exists('saved_models/'):
    os.mkdir('saved_models')
if not os.path.exists('aae_gan_images/') and PLOT:
    os.mkdir('aae_gan_images')
#-------image specific settigns----------
encoder = dc_encoder(latent_dim,imsize,ndf=nef)
generator = dc_generator(latent_dim,ngf=ngf,nc=3,imgsize=imsize)
discriminator_latent = dfc_discriminator(latent_dim)
discriminator_image = dc_discriminator(imsize,ndf=ndf,nc=3,max_depth=7)

optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=LR_enc)
optimizer_gen= torch.optim.Adam(generator.parameters(), lr=LR_gen)
optimizer_disc_latent=torch.optim.Adam(discriminator_latent.parameters(), lr=LR_disc_latent)
optimizer_disc_image=torch.optim.Adam(discriminator_image.parameters(), lr=LR_disc_image)
state_epoch=0
if load_model:
  print('reload model....')
  state_dict_enc=torch.load('saved_models/aae_gan_enc_'+str(imsize)+'.pkl')
  state_dict_gen=torch.load('saved_models/aae_gan_gen_'+str(imsize)+'.pkl')
  state_dict_disc_latent=torch.load('saved_models/aae_gan_disc_latent_'+str(imsize)+'.pkl')
  state_dict_disc_image=torch.load('saved_models/aae_gan_disc_image_'+str(imsize)+'.pkl')
  state_epoch=state_dict_gen['epoch']
  encoder.load_state_dict(state_dict_enc['model_state'])
  generator.load_state_dict(state_dict_gen['model_state'])
  discriminator_latent.load_state_dict(state_dict_disc_latent['model_state'])
  discriminator_image.load_state_dict(state_dict_disc_image['model_state'])
  optimizer_enc.load_state_dict(state_dict_enc['optimizer_state'])
  optimizer_gen.load_state_dict(state_dict_gen['optimizer_state'])
  optimizer_disc_latent.load_state_dict(state_dict_disc_latent['optimizer_state'])
  optimizer_disc_image.load_state_dict(state_dict_disc_image['optimizer_state'])

np_enc = sum(np.prod(list(p.size())) for p in encoder.parameters())
np_gen  = sum(np.prod(list(p.size())) for p in generator.parameters())
np_tot=np_enc+np_gen
print ('Number of params in dc_autoencoder: %d' % np_tot)


def disc_loss_fuction(d,d_hat,batchSize):
  loss=criterion(d,quasi_real_label[:batchSize])+criterion(d_hat,quasi_fake_label[:batchSize])
  return loss

def gen_loss_function(d_hat,batchSize):
    #loss=torch.sum(-torch.log(d_hat))/batchSize
    loss=criterion(d_hat,real_label[:batchSize])
    return loss

#---prepare noise and labels
criterion = nn.BCELoss()
real_label = torch.full((batch_size,), 1)
fake_label = torch.full((batch_size,), 0)
quasi_real_label = torch.full((batch_size,), 0.9)
quasi_fake_label = torch.full((batch_size,), 0.1)
fixed_noise = torch.randn(batch_size, latent_dim, 1, 1)
if latent_distribution=='uniform':
  fixed_noise=torch.rand(batch_size, latent_dim,1,1)-0.5
else:
  fixed_noise=torch.randn(batch_size, latent_dim,1,1)


#----- training loop--------
# batch_idx=[0]
# for b in range(int(np.floor(n_img_loaded/batch_size))):
#   batch_idx.append((b+1)*batch_size)
# if n_img_loaded>batch_size*(len(batch_idx)-1):
#   batch_idx.append(n_img_loaded)
#n_minibatches=len(batch_idx)-1
n_minibatches=len(trainloader)


i = state_epoch
def closure():  
    global i,net_input,img_torch_array
    # if shuffle and (i+1)%shuffle_every==0:

    #   shuffle_idx=torch.randperm(net_input.size()[0])
    #   if input_corruption=='holes':
    #     net_input=img_torch_array.detach().clone().masked_fill_((-img_torch_array*masked_torch_array[shuffle_idx]+1).type(torch.ByteTensor), -1)[shuffle_idx]
    #   elif input_corruption=='noise':
    #     print('new noise:')
    #     net_input=net_input[shuffle_idx]+torch.randn(n_img_loaded, 3, imsize, imsize)*0.1
    #   else:
    #     net_input=net_input[shuffle_idx]
    #   img_torch_array=img_torch_array[shuffle_idx]

    # for idx in range(len(batch_idx)-1):
    for idx, data in enumerate(trainloader, 0):
      #batchSize=batch_idx[idx+1]-batch_idx[idx]
      batchSize=batch_size
      optimizer_gen.zero_grad()
      optimizer_enc.zero_grad()
      optimizer_disc_latent.zero_grad()
      optimizer_disc_image.zero_grad()

      if latent_distribution=='uniform':
        latent_noise=torch.rand(batchSize, latent_dim,1,1)-0.5
      else:
        latent_noise=torch.randn(batchSize, latent_dim,1,1)

#-----propagate the autoencoder
      #latent_out=encoder(net_input[batch_idx[idx]:batch_idx[idx+1],:])
      latent_out=encoder(data[0]) 
      out = generator(latent_out)
      out_noise = generator(latent_noise)

#----discriminator training, in the latent and the image space
      d_latent=discriminator_latent(torch.squeeze(latent_noise))
      d_hat1_latent = discriminator_latent(torch.squeeze(latent_out).detach())
      #d_image=discriminator_image(img_torch_array[batch_idx[idx]:batch_idx[idx+1],:])
      d_image=discriminator_image(data[0])
      d_hat1_image=discriminator_image(out_noise.detach())
      loss_disc_latent = disc_loss_fuction(d_latent,d_hat1_latent,batchSize)
      loss_disc_image = disc_loss_fuction(d_image,d_hat1_image,batchSize)
      loss_disc_latent.backward()
      loss_disc_image.backward()
      optimizer_disc_latent.step()
      optimizer_disc_image.step()

#-----generator training
      d_hat2_latent = discriminator_latent(torch.squeeze(latent_out))
      d_hat2_image = discriminator_image(out)
      d_hat2_image_noise = discriminator_image(out_noise)
      #ae_loss=ae_loss_function(out,img_torch_array[batch_idx[idx]:batch_idx[idx+1],:],d_hat2_latent,d_hat2_image,batchSize)#learns a can regularized autoencoding
      #de=out-img_torch_array[batch_idx[idx]:batch_idx[idx+1],:] 
      de=out-data[0]
      recon_loss=l_reg_recon*torch.sum(torch.mul(de,de))/batchSize
      disc_loss=l_reg_latent*criterion(d_hat2_latent,real_label[:batchSize])+l_reg_image*criterion(d_hat2_image,real_label[:batchSize])
      ae_loss=recon_loss+disc_loss
      ae_loss.backward()
      gen_loss=gen_loss_function(d_hat2_image_noise,batchSize)#learns generation from noise
      gen_loss.backward()
      optimizer_enc.step()
      optimizer_gen.step()


      print('[%d/%d][%d/%d]  Recon_loss: %.4f D(x): %.4f D(G(z)): %.4f D(E(x)): %.4f'
            % (i+1, num_iter+state_epoch, idx+1, n_minibatches,
              recon_loss.item(), d_image.mean().item(), d_hat1_image.mean().item(), d_hat1_latent.mean().item()))

    # if  (i+1) % show_every == 0:
      if (idx+1) % show_every_n_batch == 0:
        print('sample from latent prior..')

        # vutils.save_image(img_torch_array[batch_idx[idx]:batch_idx[idx+1],:],
        #         '%s/real_samples.png' % 'aae_gan_images',
        #         normalize=True)
        vutils.save_image(data[0],
                '%s/real_samples.png' % 'aae_gan_images',
                normalize=True)
        with torch.no_grad():
          fake = generator(fixed_noise)
          vutils.save_image(fake.detach(),
                  '%s/fake_samples_epoch_%03d_%03d.png' % ('aae_gan_images', i,idx),
                  normalize=True)
        if PLOT:
          #idx=n_minibatches-2
          for n_im in range(np.minimum(5,batch_size)):
            out_np = torch.clamp(out[n_im,:].transpose(0,1).transpose(1,2).detach(),0,1).numpy()
            # corrupted_np = torch.clamp(net_input[batch_idx[idx]+n_im,:].transpose(0,1).transpose(1,2).detach(),0,1).numpy()
            # orig_np=img_torch_array[batch_idx[idx]+n_im,:].transpose(0,1).transpose(1,2).detach().numpy()
            corrupted_np = torch.clamp(data[0][n_im,:].transpose(0,1).transpose(1,2).detach(),0,1).numpy()
            orig_np=data[0][n_im,:].transpose(0,1).transpose(1,2).detach().numpy()
            save_comparison_plot(corrupted_np,out_np,orig_np,'aae_gan_images/'+str(i)+'_'+str(idx)+'_'+str(n_im))

        if latent_dim==2:
          np_latent_out=torch.squeeze(latent_out).detach().numpy()
          np_latent_noise=torch.squeeze(latent_noise).detach().numpy()
          plt.clf()
          plt.plot(np_latent_out[:,0], np_latent_out[:,1], 'ro',np_latent_noise[:,0],np_latent_noise[:,1],'b*')
          plt.savefig('aae_gan_images/latent_space_'+str(i))

    if (i+1)%save_every==0:
      print('save model ...')
      torch.save({'epoch': i, 'model_state': encoder.state_dict(),'optimizer_state': optimizer_enc.state_dict()}, 'saved_models/aae_gan_enc_'+str(imsize)+'.pkl')
      torch.save({'epoch': i, 'model_state': generator.state_dict(),'optimizer_state': optimizer_gen.state_dict()}, 'saved_models/aae_gan_gen_'+str(imsize)+'.pkl')
      torch.save({'epoch': i, 'model_state': discriminator_latent.state_dict(),'optimizer_state': optimizer_disc_latent.state_dict()}, 'saved_models/aae_gan_disc_latent_'+str(imsize)+'.pkl')
      torch.save({'epoch': i, 'model_state': discriminator_image.state_dict(),'optimizer_state': optimizer_disc_image.state_dict()}, 'saved_models/aae_gan_disc_image_'+str(imsize)+'.pkl')      
    i += 1


#-----call optimizer and save stuff ----
if TRAINING:
  print('start training ...')
  for j in range(num_iter):
    closure()
  torch.save({'epoch': i, 'model_state': generator.state_dict(),'optimizer_state': optimizer_gen.state_dict()}, 'saved_models/aae_gan_gen_'+str(imsize)+'.pkl')
  torch.save({'epoch': i, 'model_state': encoder.state_dict(),'optimizer_state': optimizer_enc.state_dict()}, 'saved_models/aae_gan_enc_'+str(imsize)+'.pkl')
  torch.save({'epoch': i, 'model_state': discriminator_latent.state_dict(),'optimizer_state': optimizer_disc_latent.state_dict()}, 'saved_models/aae_gan_disc_latent_'+str(imsize)+'.pkl')  
  torch.save({'epoch': i, 'model_state': discriminator_image.state_dict(),'optimizer_state': optimizer_disc_image.state_dict()}, 'saved_models/aae_gan_disc_image_'+str(imsize)+'.pkl')  




