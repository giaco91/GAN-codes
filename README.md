# GAN-codes
The aim of this repo is to build my own deep learning library of example implementation in pytorch most of which have a GAN component.

## Deep image prior (DIP)
The file "dip.py" is a vanilla implementation for inpaining that uses the deep image prior proposed by Dmitry Ulyanov et. al.

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
1. a mode regularized GAN (MD-GAN), Tong Che et. al.
2. a adversarial autoencoder (AAE),  Alireza Makhzani et. al.

To the best of my knowledge the AAE-GAN in this form has not been proposed yet. 

### Description
Our AAE-GAN consists of an encoder E and a generator G that together form an autoencoder. The autoencoder tries to minimize the reconstruction error ||X-G(E(X))||^2. There are two additional regularization terms. First, the encoder is regularized by a discriminator D1 that discriminates between the distribution E(X) and a prior distribution p(Z) in the latent space. Second, the generator is regularized by a discriminator D2 that discriminates between the distribution G(E(X)) and the true data generating distribution P(X). In parallel to that, we train the generator seperately in a GAN setup, where we use the discriminator D2 to discriminate between G(Z) (Z distributed as P(Z)) and the true data distribution P(X).

Since all basic networks (such as encoder, generator and discriminator) used in this repo are identical, we can directly compaire the performance of the AAE-GAN to the other models under similar architectures (and similar implementation style).
