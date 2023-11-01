# Convolutional Generative Adversarial Network for simple Images

## Content
This folder contains the different code and files along with a notebook explaining the process and showing some results of a simple DCGAN.

## Files

- [main.py](#Main)
- [get_loader.py](#Get_loader)
- [model.py](#Model)
- [utils.py](#Utils)
- [DCGAN.ipynb](#Notebook_DCGAN)

### Main

This file is meant to be executed if the objective is to train a network, it contains:
- ***Training loop***: Training loop in order to train the network.
- ***Variable definition***: All the variables such as the channels of the images, batch size, size of the images, learning rate, number of epochs... are defined in this file.

### Get_loader

In this file the images are downloaded and the dataloader is created with a single function  

### Model

File containing the whole model for the task. The model is divided in 2 sections:
- `Discriminator`: The Discriminator function is to take an image and tell apart between images from the real dataset or images generated from the generator.
- `Generator`: The Generator function is to generate an image taking a hidden space as a base and try to convince the discriminator its image is from the original dataset.

### Utils

In the utils file we have 3 different functions:
- `weight_init`: This function inisialize the weights following a normal distribution.
- `plot_losses`: This function plots the losses for both sections of the model, generator and discriminator.
- `animations`: This function shows the animation on how the generator changes its generations trough the different epochs.

### Notebook_DCGAN

Python notebook where all the process is explained and where people can give a try to the model without a lot of complications.


