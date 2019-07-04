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
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from common_utils import *

#---global settings

dtype = torch.FloatTensor

TRAINING = True
imsize = 32
load_model = True
num_iter =10
print_every_n_batch=2000
shuffle=True
save_every=1
batch_size=10#must be smaller or equal than max_num_img
n_classes=10
LR=0.0005
ndf=5
max_depth=7

TEST=True
test_every=1

num_workers=2
#----- specifiy the figure
random.seed(1)
torch.manual_seed(1)

if not os.path.exists('saved_models/'):
    os.mkdir('saved_models')

root_to_dataset='/Users/Giaco/Documents/Elektrotechnik-Master/image_processing/my_deep_image_prior/data/'
trainset=dset.CIFAR10(root=root_to_dataset, train=True, transform=transforms.Compose([
                                   transforms.Resize(imsize),
                                   transforms.CenterCrop(imsize),
                                   transforms.ToTensor(),
                               ]), download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
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


#-------load model----------
mult_discriminator=dc_mult_discriminator(imsize,ndf=ndf,n_classes=n_classes,nc=3,max_depth=max_depth)
optimizer_mult_disc= torch.optim.Adam(mult_discriminator.parameters(), lr=LR)
state_epoch=0
if load_model:
  print('reload model....')
  state_dict_mult_disc=torch.load('saved_models/classifier_mult_discriminator'+str(imsize)+'.pkl')
  mult_discriminator.load_state_dict(state_dict_mult_disc['model_state'])
  optimizer_mult_disc.load_state_dict(state_dict_mult_disc['optimizer_state'])
  state_epoch=state_dict_mult_disc['epoch']



#------
np= sum(np.prod(list(p.size())) for p in mult_discriminator.parameters())
print ('Number of params in dc_classifier: %d' % np)


criterion = nn.CrossEntropyLoss()
i = state_epoch
def closure():  
    global i

    running_loss = 0.0
    rl=0
    for j, data in enumerate(trainloader, 0):
      optimizer_mult_disc.zero_grad()

# #----discriminator between generators
      likelihood=mult_discriminator(data[0])
      loss_mult_disc=criterion(likelihood,data[1])
      loss_mult_disc.backward()
      optimizer_mult_disc.step()
      running_loss += loss_mult_disc.item()
      rl+=1
      # print statistics
      if j % print_every_n_batch == print_every_n_batch-1 or j==len(trainloader)-1:    # print every 200 mini-batches
          print('[%d/%d][%d/%d] loss: %.3f' %
                (i+1, num_iter+state_epoch, j+1, len(trainloader),running_loss / rl))
          running_loss = 0.0
          rl=0

    if (i+1)%save_every==0:
      print('save model ...')
      torch.save({'epoch': i, 'model_state': mult_discriminator.state_dict(),'optimizer_state': optimizer_mult_disc.state_dict()}, 'saved_models/classifier_mult_discriminator'+str(imsize)+'.pkl') 
    
    if TEST and (i+1)%test_every==0:

      correct = 0
      total = 0
      with torch.no_grad():
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
    i += 1

    return

#-----call optimizer and save stuff ----
if TRAINING:
  print('start training ...')
  for j in range(num_iter):
    closure()
   
  torch.save({'epoch': i, 'model_state': mult_discriminator.state_dict(),'optimizer_state': optimizer_mult_disc.state_dict()}, 'saved_models/classifier_mult_discriminator'+str(imsize)+'.pkl')

if TEST and (i+1)%test_every==0:
    # # get some random training images
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    # # show images
    # plt.imshow(torchvision.utils.make_grid(images).permute(1, 2, 0))
    # plt.show()
    # #print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    # dataiter = iter(testloader)
    # images, labels = dataiter.next()
    # plt.imshow(vutils.make_grid(images).permute(1, 2, 0))
    # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
    # plt.show()
    # outputs = mult_discriminator(images)
    # print(outputs)
    # _, predicted = torch.max(outputs, 1)

    # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
    #                             for j in range(batch_size)))


with torch.no_grad():
    correct = 0
    total = 0
    for data in trainloader:
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


