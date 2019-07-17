import matplotlib.pyplot as plt
import glob
import time
import os
import random
import numpy as np
from models.dc_generator import dc_root_generator,dc_final_generator
from models.dc_discriminator import dc_discriminator
from models.dc_mult_discriminator import dc_mult_discriminator
import torch
import torch.optim
import time
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms

from common_utils import *

#---global settings

dtype = torch.FloatTensor
# PLOT = True
TRAINING = True
imsize = 32
load_model = False
num_iter =1
show_every_n_batch=1
shuffle=True
save_every=1
batch_size=100#must be smaller or equal than max_num_img
LR_gen= 0.0005
LR_disc=0.0005
nz=60
ngf=40
nmdf=20
label_noise=0
max_n_channel_gen=1000
max_n_channel_disc=1000
max_depth_disc=6

num_workers=2
#----- specifiy the figure
random.seed(1)
torch.manual_seed(1)

# path_to_mult_discriminator='saved_models/classifier_mult_discriminator'+str(imsize)+'.pkl'

root_to_dataset='/Users/Giaco/Documents/Elektrotechnik-Master/image_processing/my_deep_image_prior/data/'
dataset=dset.CIFAR10(root=root_to_dataset, train=True, transform=transforms.Compose([
                                   transforms.Resize(imsize),
                                   transforms.CenterCrop(imsize),
                                   transforms.ToTensor(),
                                   
                               ]), download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)

testset=dset.CIFAR10(root=root_to_dataset, train=False, transform=transforms.Compose([
                                   transforms.Resize(imsize),
                                   transforms.CenterCrop(imsize),
                                   transforms.ToTensor(),
                               ]), download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


n_classes=10
r_batch_size=int(batch_size/n_classes)
batch_size=int(r_batch_size*n_classes)

#----torch inizializations
if not os.path.exists('saved_models/'):
    os.mkdir('saved_models')
if not os.path.exists('d_gan_images/') and PLOT:
    os.mkdir('d_gan_images/')
#-------image specific settigns----------
root_generator=dc_root_generator(nz,ngf=ngf,nc=3,imgsize=imsize,max_n_channel=max_n_channel_gen)
optimizer_root_gen=torch.optim.Adam(root_generator.parameters(), lr=LR_gen)
generator_list=[]
optimizer_gen_list=[]
for g in range(n_classes):
 generator_list.append(dc_final_generator(root_generator.in_ch,input_padding=root_generator.input_padding,output_padding=root_generator.output_padding,imgsize=imsize))
 optimizer_gen_list.append(torch.optim.Adam(generator_list[g].parameters(), lr=LR_gen*n_classes))#we make the lr of the final generator larger due to the implementation of the loss
mult_discriminator=dc_mult_discriminator(imsize,ndf=nmdf,n_classes=n_classes+1,nc=3,max_depth=max_depth_disc,softmax=False,max_n_channel=max_n_channel_disc)
optimizer_mult_disc= torch.optim.Adam(mult_discriminator.parameters(), lr=LR_disc)

state_epoch=0
if load_model:
  print('reload model....')
  for g in range(n_classes):
    state_dict_gen=torch.load('saved_models/d_gan_final_generator'+str(imsize)+'_'+str(g)+'.pkl')
    generator_list[g].load_state_dict(state_dict_gen['model_state'])
    optimizer_gen_list[g].load_state_dict(state_dict_gen['optimizer_state'])
  state_dict_root_gen=torch.load('saved_models/d_gan_root_generator'+str(imsize)+'.pkl')
  state_epoch=state_dict_root_gen['epoch']
  root_generator.load_state_dict(state_dict_root_gen['model_state'])
  optimizer_root_gen.load_state_dict(state_dict_root_gen['optimizer_state'])
  state_dict_mult_disc=torch.load('saved_models/d_gan_mult_discriminator'+str(imsize)+'.pkl')
  mult_discriminator.load_state_dict(state_dict_mult_disc['model_state'])
  optimizer_mult_disc.load_state_dict(state_dict_mult_disc['optimizer_state'])

np_gen  = sum(np.prod(list(p.size())) for p in root_generator.parameters())
for g in range(n_classes):
  np_gen+=sum(np.prod(list(p.size())) for p in generator_list[g].parameters())
print ('Number of params in d_generator %d' % np_gen)
np_disc=sum(np.prod(list(p.size())) for p in mult_discriminator.parameters())
print('Number of params in discriminator %d' % np_disc)



criterion_CE = nn.CrossEntropyLoss()
fixed_noise = torch.randn(batch_size, nz, 1, 1)
real_label = torch.full((batch_size,), 1)
fake_label = torch.full((batch_size,), 0)

softmax=torch.nn.Softmax(dim=1)

#unbias_fact=1/(n_classes**2/4+n_classes/2-1)#note for the special case of the classical gan (n_classes=2) we have unbias_fact=1
epsilon=1e-10
def disc_fake_loss(d_hat,label):
    props=softmax(d_hat)
    loss_overall=-torch.sum(torch.log(props[:,-1]+epsilon))
    loss_class=-torch.sum(torch.log(props[:,-1]/(props[:,-1]+props[:,label])+epsilon))
    return (loss_overall+loss_class)/batch_size


def generator_loss(d_hat,label):
  props=softmax(d_hat)
  loss_overall=-torch.sum(torch.log(1-props[:,-1]+epsilon))
  loss_class=-torch.sum(torch.log(props[:,label]/(props[:,-1]+props[:,label])+epsilon))
  return (loss_overall+loss_class)/batch_size


i = state_epoch
def closure():  
    global i,net_input,img_torch_array

    for j, data in enumerate(dataloader, 0):
      start = time.time()
      optimizer_mult_disc.zero_grad()
      optimizer_root_gen.zero_grad()

#----label noise to weaken the multiclass distriminator
      if label_noise>0:
        n_s=int(np.floor(label_noise*batch_size))
        shuffle_idx=torch.randperm(n_s)
        data[1][0:n_s]=data[1][0:n_s][shuffle_idx]

#-----generation
      # noise = torch.randn(r_batch_size, nz, 1, 1)
      noise = torch.randn(batch_size, nz, 1, 1)
      out_root=root_generator(noise)
      out_list=[]
      for g in range(n_classes):
        out_list.append(generator_list[g](out_root))
        # out_list.append(generator_list[g](out_root[g*r_batch_size:(g+1)*r_batch_size]))
 
#----discriminator between real and fake images training
      d=mult_discriminator(data[0])
      loss_disc_real=criterion_CE(d,data[1])
      loss_disc_real.backward()
      loss_disc_fake=0
      for g in range(n_classes):
        d_hat1=mult_discriminator(out_list[g][:r_batch_size].detach())
        loss_disc_fake+=disc_fake_loss(d_hat1,g)
      # loss_disc_fake/=n_classes
      loss_disc_fake.backward()
      optimizer_mult_disc.step()

      ##---test if mult_discriminator actually is informative (i.e. if it works)
      _, predicted = torch.max(d, 1)
      print('classification probability: '+str((predicted == data[1]).sum().item()/batch_size*100)+'%')

#---- generator training
      loss_gen=0
      for g in range(n_classes):
        optimizer_gen_list[g].zero_grad()
        d_hat=mult_discriminator(out_list[g])
        loss_gen+=generator_loss(d_hat,g)
      loss_gen/=n_classes#we need to average over the generators
      loss_gen.backward()
      optimizer_root_gen.step()
      for g in range(n_classes):
        optimizer_gen_list[g].step()
      
      print('[%d/%d][%d/%d]  loss_disc_real_class: %.4f loss_disc_fake: %.4f loss_gen: %.4f'
            % (i+1, num_iter+state_epoch, j+1, len(dataloader),
                loss_disc_real.item(), loss_disc_fake.item(), loss_gen.item()))

      if (j+1) % show_every_n_batch == 0:
          print('sample mini batch...')
          vutils.save_image(data[0],
                  '%s/real_samples.png' % 'd_gan_images',
                  normalize=True)
          with torch.no_grad():
            out_root=root_generator(fixed_noise)
            for g in range(n_classes):
              fake = generator_list[g](out_root)
              vutils.save_image(fake.detach(),
                      '%s/fakes_%03d_%03d_generator_%01d.png' % ('d_gan_images', i+1,j+1,g),
                      normalize=True)
      end = time.time()
      print('batch time: '+str(end-start))

    if (i+1)%save_every==0:
      print('save model ...')
      for g in range(n_classes):
        torch.save({'epoch': i, 'model_state': generator_list[g].state_dict(), 'optimizer_state': optimizer_gen_list[g].state_dict()}, 'saved_models/d_gan_final_generator'+str(imsize)+'_'+str(g)+'.pkl') 
      torch.save({'epoch': i, 'model_state': root_generator.state_dict(),'optimizer_state': optimizer_root_gen.state_dict()}, 'saved_models/d_gan_root_generator'+str(imsize)+'.pkl')    
      torch.save({'epoch': i, 'model_state': mult_discriminator.state_dict(),'optimizer_state': optimizer_mult_disc.state_dict()}, 'saved_models/d_gan_mult_discriminator'+str(imsize)+'.pkl')    
    i += 1



    return

#-----call optimizer and save stuff ----
if TRAINING:
  print('start training ...')
  for j in range(num_iter):
    closure()
  for g in range(n_classes):
    torch.save({'epoch': i, 'model_state': generator_list[g].state_dict(), 'optimizer_state': optimizer_gen_list[g].state_dict()}, 'saved_models/d_gan_final_generator'+str(imsize)+'_'+str(g)+'.pkl') 
  torch.save({'epoch': i, 'model_state': root_generator.state_dict(),'optimizer_state': optimizer_root_gen.state_dict()}, 'saved_models/d_gan_root_generator'+str(imsize)+'.pkl')
  torch.save({'epoch': i, 'model_state': mult_discriminator.state_dict(),'optimizer_state': optimizer_mult_disc.state_dict()}, 'saved_models/d_gan_mult_discriminator'+str(imsize)+'.pkl') 

else:
  dataiter = iter(dataloader)
  images, labels = dataiter.next()
  vutils.save_image(images,'d_gan_images/real_samples'+str(imsize)+'.png',normalize=True)


with torch.no_grad():
    correct = 0
    total = 0
    for data in dataloader:
        images, labels = data
        outputs = mult_discriminator(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 train images: %d %%' % (
        100 * correct / total))
    correct=0
    total=0  
    for data in testloader:
        images, labels = data
        outputs = mult_discriminator(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

class_correct = list(0. for i in range(10))

class_total = list(0. for i in range(10))
with torch.no_grad():
  for data in testloader:
      images, labels = data
      outputs = mult_discriminator(images)
      _, predicted = torch.max(outputs, 1)
      c = (predicted == labels).squeeze()
      for i in range(4):
          label = labels[i]
          class_correct[label] += c[i].item()
          class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


mult_discriminator_val=dc_mult_discriminator(imsize,ndf=5,n_classes=n_classes,nc=3,max_depth=7)
if load_model:
  print('load validation discriminator model....')
  state_dict_mult_disc_val=torch.load('saved_models/classifier_mult_discriminator'+str(imsize)+'.pkl')
  mult_discriminator_val.load_state_dict(state_dict_mult_disc_val['model_state'])

with torch.no_grad():
  correct = 0
  correct_val=0
  total = 0
  N=1000
  noise = torch.randn(1000, nz, 1, 1)
  out_root=root_generator(noise)
  for g in range(n_classes):
    out_g=generator_list[g](out_root)
    d_g=mult_discriminator(out_g)
    d_g_val=mult_discriminator_val(out_g)
    _, predicted = torch.max(d_g,1)
    _, predicted_val = torch.max(d_g_val,1)
    total += predicted.size(0)
    correct += (predicted == g).sum().item()
    correct_val += (predicted_val == g).sum().item()

  print('Accuracy of d_gan according to the adversary discriminator: %d %%' % (
      100 * correct / total))
  print('Accuracy of d_gan according to the validation discriminator: %d %%' % (
      100 * correct_val / total))















