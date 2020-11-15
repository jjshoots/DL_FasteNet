import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np
from numpy import random as nprand

class ImageLoader(Dataset):
    def __init__(self, raw_images, ground_truths, image_size, crops_per_image=1000, crop_quant=8, crop_size=32):
        self.crops_per_image = crops_per_image
        self.crop_size = crop_size
        self.crop_dim = crop_quant * crop_size

        self.raw_images = raw_images
        self.ground_truths = ground_truths
        self.image_size_x, self.image_size_y = image_size

        self.number_of_images = len(raw_images)

        print(f'Dataloader initiated: crop size={self.crop_dim}, number of images={self.number_of_images}, crops per image={self.crops_per_image}')

    def __len__(self):
        return self.number_of_images * self.crops_per_image

    def __getitem__(self, idx):
        # get random image and gorund truth image
        index = int(idx / self.crops_per_image)
        image = self.raw_images[index]
        truth = self.ground_truths[index]

        # get the limit of the starting index that we can crop
        x_lim = self.image_size_x - self.crop_dim
        y_lim = self.image_size_y - self.crop_dim

        # get a random start index and corresponding end index
        x_start = nprand.randint(0, x_lim)
        y_start = nprand.randint(0, y_lim)
        x_end = x_start + self.crop_dim
        y_end = y_start + self.crop_dim

        # crop the image to be our data
        data = image[x_start:x_end, y_start:y_end].unsqueeze(0)
        label = truth[x_start:x_end, y_start:y_end].unsqueeze(0).unsqueeze(0)
        label = F.interpolate(label, size=[self.crop_size, self.crop_size]).squeeze(0)

        return data.detach(), label.detach()