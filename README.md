# GAN-codes
The aim of this repo is to build my own deep learning library of example implementation in pytorch most of which have a GAN component.

## Deep image prior (DIP)
The file "dip.py" is a vanilla implementation for inpaining that uses the deep image prior only.

## Deep convolutional autoencoder (AE)
The file "dc_ae.py" is a vanilla implementation for a deep convolutional autoencoder.

## Deep convolutional GAN (DC-GAN)
The file "dc_gan.py" is a vanilla implementation of a deep convolutional GAN.

## Deep convolutional evolutionary GAN (E-GAN)
The file "e_gan.py" is a vanilla implementation of a deep convolutional evolutionary GAN. It is highly inspired by the ideas of Chaoyue Wang et. al.

## Deep convolutional adversarial autoencoder (AAE)
The file "aae.py" is a vanilla implementation of a deep convolutional adversarial autoencoder.

## DC-GAN regularized by a AAE (AAE-GAN)
The file "aae_gan.py" is a vanilla implementation of a AAE regularized GAN. It is a mixture between the ideas of 
1. a mode regularized GAN (MD-GAN), Tong Che et. al.
2. a adversarial autoencoder (AAE),  Alireza Makhzani et. al.
3. a variational audoencoder with a GAN regularizer (VAE-GAN),  Anders Larsen et. al.

To the best of my knowledge this has not been proposed yet. Since all basic networks (such as encoder, generator and discriminator) used in this repo are identical, we can directly compaire the performance of the AAE-GAN to the other models under similar architectures (and similar implementation style).
