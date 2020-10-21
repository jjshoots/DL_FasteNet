#!/usr/bin/env python3
import time
import os
import numpy as np
import random
from numpy.lib.type_check import imag

import torch
from torch._C import Size
import torch.nn as nn
import torch.nn.functional as func
import torchvision.transforms.functional as TF

import imageio
from PIL import Image
import matplotlib.pyplot as plt

from ImageLoader import ImageLoader

# params
DIRECTORY = os.path.dirname(__file__)
BATCH_SIZE = 64


images = []
truths = []

for i in range(3):
    image_path = os.path.join(DIRECTORY, f'img/raw{i+1}.png')
    truth_path = os.path.join(DIRECTORY, f'img/ground{i+1}.png')

    image = TF.to_tensor(Image.open(image_path))[0, :, :]
    truth = TF.to_tensor(Image.open(truth_path))[0, :, :]
    truth /= torch.max(truth)
    
    images.append(image)
    truths.append(truth)

dataset = ImageLoader(images, truths, (images[0].shape[0], images[0].shape[1]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

if 1:
    dataiter = iter(dataloader)
    user_input = None
    while user_input != 'Y':
        user_input = input('Key in "Y" to end display, enter to continue...')
        data, label = dataiter.next()

        plt.imshow(data[0].squeeze())
        plt.show()
        plt.imshow(label[0].squeeze())
        plt.show()