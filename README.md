# minidiffusion
Diffusion model in a few lines of code. Code explained in detail.


This code is mainly for my own learning process of all the details of a diffusion model. 
It flows the paper https://arxiv.org/pdf/2006.11239.pdf "Denoising Diffusion Probabilistic Models" and I tried to ignore anything that is not related to the paper.

The code has different parts:

* ddpm.py: this is the main file. It contains the Diffusion class that is used to train the moddel.
           The training is done in the train() function.
* modules.py: the modules used in the model.
  * EMA: exponential moving average
  * DoubleConv: double convolutional layer
  * SelfAttention: self attention layer
  * Down: downsampling layer
  * Up: upsampling layer
  * UNet: U-Net with self attention
    * The encoder is a 256 channel convolutional layer
    * The bottleneck is a 256 channel convolutional layer
    * The decoder is a 256 channel convolutional layer

The trainig process will save: 
1) chkpt files while training
2) some generated images for checking the evolution

To-Do:
* Add WandB support for logging and checking the evolution of the model during training
* Add conditional diffusion so we can train on images with labels


