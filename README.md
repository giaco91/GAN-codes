# GAN-codes
The aim of this repo is to build my own deep learning library of example implementation in pytorch most of which have a GAN component.

## Deep image prior (DIP)
The file "dip.py" is a vanilla implementation for inpaining that uses the deep image prior proposed by Dmitry Ulyanov et. al.

## Deep convolutional multiclass discriminator (DC-MD)
The file classifier.py is a vanilla implementation of a deep convolutional mutliclass discriminator. We will use the multiclass discriminator for my MC-GAN's.

## Deep convolutional autoencoder (AE)
The file "dc_ae.py" is a vanilla implementation for a deep convolutional autoencoder.

## Deep convolutional GAN (DC-GAN)
The file "dc_gan.py" is a vanilla implementation of a deep convolutional, proposed by Alec Radford & Luke Metz et. al.

## Deep convolutional evolutionary GAN (E-GAN)
The file "e_gan.py" is a vanilla implementation of a deep convolutional evolutionary GAN. It is highly inspired by the ideas of Chaoyue Wang et. al.

## Deep convolutional adversarial autoencoder (AAE)
The file "aae.py" is a vanilla implementation of a deep convolutional adversarial autoencoder, proposed by Alireza Makhzani et. al.

## DC-GAN regularized by a AAE (AAE-GAN)
The file "aae_gan.py" is a vanilla implementation of a AAE regularized GAN. It is a mixture between 
1. Mode regularized GAN (MD-GAN), Tong Che et. al.
2. Adversarial Autoencoder (AAE),  Alireza Makhzani et. al.

To the best of my knowledge the AAE-GAN in this form has not been proposed yet. 

### Description
Our AAE-GAN consists of an encoder E and a generator G that together form an autoencoder. The autoencoder tries to minimize the reconstruction error ||X-G(E(X))||^2. There are two additional regularization terms. First, the encoder is regularized by a discriminator D1 that discriminates between the distribution E(X) and a prior distribution p(Z) in the latent space. Second, the generator is regularized by a discriminator D2 that discriminates between the distribution G(E(X)) and the true data generating distribution P(X). In parallel to that, we train the generator seperately in a GAN setup, where we use the discriminator D2 to discriminate between G(Z) (Z distributed as P(Z)) and the true data distribution P(X).

Since all basic networks (such as encoder, generator and discriminator) used in this repo are identical, we can directly compaire the performance of the AAE-GAN to the other models under similar architectures (and similar implementation style).

## Deep convolutional multiclass GAN (MC-GAN)

The file "d_gan.py" (name will be adjusted to mc_gan.py) implements a vanilla implementation of a MC-GAN. The model assumes that we have additional class-label information of the data. Let K be the amount of class-labels in the data set. The labels will be used to help against the problem of mode collapse.

To the best of my knowledge the MC-GAN has not been proposed yet. 

### Description
Discriminator:
The discriminator of the MC-GAN is a multiclass discriminator that learns to classify its input in to K+1 classes, namely the K classes in which the data set is partitioned and the K+1 class which is implied by the fake samples from the generator. 

Generator:
While the generator (as usual) consists of serveral upsampling steps. However, the last upsampling step (to the shape of the data images) is learned by K different sub-networks, one for each class. We can then think of K different generators g=1,2,...,K with a large amount of shared parameteres and a shared latent space. Every generator tries has two objectives, a global and a specific one. The global objective can be seen as the standard GAN-loss function as it tries make the generated images look realistic. However, the specific objective can be seen as regularizer that enforces the shared upsampling network to be riche in information which helps to overcome the mode collapse. 
1. The global objective is to fool the mc-discriminator in terms of not beeing classified in to the (K+1)-th fake class which we can achieve by the loss function: l_global(x)=-log(1-P(class(x)=K+1)), where P(class(x)=K+1) is the probability that the image x belongs to the class K+1 according to the mc-discriminator. Note that this loss is the same for all generators g.
2. The specific objective is to fool the mc-discriminator such that it thinks that the fake image x created by the generator g belongs to class g. This can be achieved by minimizing the following loss for generator g: l_specific(x;g)=-log(P(class(x)=g)).

Since the generator has a lot of shared upsampling parameters which must be diverse in the sense that they have to act as features for different classes, we can expect that mode collapse is positiviely regularized by the specific loss function.
