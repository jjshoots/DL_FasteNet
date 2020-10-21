import torch
from torch.utils.data import Dataset

import numpy as np
from numpy import random as nprand
import random

class ImageLoader(Dataset):
    def __init__(self, raw_images, ground_truths, image_size, crops_per_image=100000, crop_quant=8, crop_size=20):
        self.crops_per_image = crops_per_image
        self.crop_size = crop_quant * crop_size

        self.raw_images = raw_images
        self.ground_truths = ground_truths
        self.image_size_x, self.image_size_y = image_size

    def __len__(self):
        return self.crops_per_image

    def __getitem__(self, idx):
        # get random image and gorund truth image
        index = random.choice(np.arange(len(self.raw_images)))
        image = self.raw_images[index]
        truth = self.ground_truths[index]

        # get the limit of the starting index that we can crop
        x_lim = self.image_size_x - self.crop_size
        y_lim = self.image_size_y - self.crop_size

        # get a random start index and corresponding end index
        x_start = nprand.randint(0, x_lim)
        y_start = nprand.randint(0, y_lim)
        x_end = x_start + self.crop_size
        y_end = y_start + self.crop_size

        # crop the image to be our data
        data = image[x_start:x_end, y_start:y_end].unsqueeze(0)
        label = truth[x_start:x_end, y_start:y_end].unsqueeze(0)

        return data, label