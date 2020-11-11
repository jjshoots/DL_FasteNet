#!/usr/bin/env python3
import time
import os
import numpy as np

import torch
from torch._C import Size
import torch.nn as nn
import torch.nn.functional as func
import torchvision.transforms.functional as TF

import imageio
from PIL import Image
import matplotlib.pyplot as plt

# params
dirname = os.path.dirname(__file__)

for i in range(3):
    from_path = os.path.join(dirname, f'img/raw{i+1}.png')
    to_path = os.path.join(dirname, f'img/test{i+1}.png')
    
    img = imageio.imread(from_path)
    img[:, :, 1:3] = 0
    imageio.imsave(to_path, img)