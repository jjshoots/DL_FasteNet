#!/usr/bin/env python3
from pickle import MARK
from sys import version
import time
import os
import sys

import cv2
import numpy as np
from numpy import random as nprand
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF

from helpers import helpers
from ImageLoader import ImageLoader
from FasteNet_Net_v2 import FasteNet_v2
from FasteNet_Net import FasteNet
from FasteNet_Net_Lite import FasteNet_Lite
from FasteNet_Net_HyperLite import FasteNet_HyperLite

# params
DIRECTORY = 'C:\AI\DATA' # os.path.dirname(__file__)
DIRECTORY2 = 'C:\AI\WEIGHTS'

VERSION_NUMBER = 5
MARK_NUMBER = 400

BATCH_SIZE = 200
NUMBER_OF_IMAGES = 700
NUMBER_OF_CYCLES = 5

# instantiate helper object
helpers = helpers(mark_number=MARK_NUMBER, version_number=VERSION_NUMBER, weights_location=DIRECTORY2)
device = helpers.get_device()

def generate_dataloader(index):
    # holder for images and groundtruths and lowest running loss
    images = []
    truths = []

    from_image = int(index * NUMBER_OF_IMAGES/NUMBER_OF_CYCLES)
    to_image = int((index+1) * NUMBER_OF_IMAGES/NUMBER_OF_CYCLES)

    # read in the images
    for i in range(from_image, to_image):
        image_path = os.path.join(DIRECTORY, f'Dataset/image/image_{i}.png')
        truth_path = os.path.join(DIRECTORY, f'Dataset/label/label_{i}.png')

        # read images
        image = TF.to_tensor(cv2.imread(image_path))[0]
        truth = TF.to_tensor(cv2.imread(truth_path))[0]
        
        # normalize inputs, 1e-6 for stability as some images don't have truth masks (no fasteners)
        image /= torch.max(image + 1e-6)
        truth /= torch.max(truth + 1e-6)

        images.append(image)
        truths.append(truth)

    print(f'Loaded images {from_image} to {to_image}')


    # feed data to the ImageLoader and start the dataloader to generate batches
    dataset = ImageLoader(images, truths, (images[0].shape[0], images[0].shape[1]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    return dataloader

# uncomment this to one to view the output of the dataset
# helpers.peek_dataset(dataloader=dataloader)

# set up net
FasteNet = FasteNet_v2().to(device)

# get latest weight file
weights_file = helpers.get_latest_weight_file()
if weights_file != -1:
    FasteNet.load_state_dict(torch.load(weights_file))

# set up loss function and optimizer
loss_function = nn.MSELoss()
optimizer = optim.SGD(FasteNet.parameters(), lr=0.001, momentum=0.9)

#  start training
for epoch in range(10000):
    if epoch % NUMBER_OF_CYCLES == 0:
        continue

    dataloader = generate_dataloader(epoch % NUMBER_OF_CYCLES)

    helpers.reset_running_loss()

    for i, data in enumerate(dataloader):
        # pull data out from dataloader
        data, labels = data[0].to(device), data[1].to(device)

        # zero the graph gradient
        FasteNet.zero_grad()

        # get the output and calculate the loss
        output = FasteNet.forward(data)

        if 0:
            plt.imshow(data[0].squeeze().to('cpu').detach().numpy())
            plt.show()
            plt.imshow(labels[0].squeeze().to('cpu').detach().numpy())
            plt.show()
            plt.imshow(output[0].squeeze().to('cpu').detach().numpy())
            plt.show()

        # backprop
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()

        weights_file = helpers.training_checkpoint(loss=loss, iterations=i, epoch=epoch)

        if weights_file != -1:
            torch.save(FasteNet.state_dict(), weights_file)
        




# FOR INFERENCING
# FOR INFERENCING
# FOR INFERENCING

# set frames to render > 0 to perform inference
torch.no_grad()
FasteNet.eval()
frames_to_render = 0
start_time = time.time()

# set to true for inference
for _ in range(frames_to_render):
    input = images[nprand.randint(0, NUMBER_OF_IMAGES)].unsqueeze(0).unsqueeze(0).to(device)[..., :1600]
    saliency_map = FasteNet.forward(input)
    torch.cuda.synchronize()

    # draw contours on original image
    contour_image, contour_number = helpers.saliency_to_contour(input=saliency_map, original_image=input, fastener_area_threshold=5, input_output_ratio=8)

    # set to true to display images
    if 1:
        comparison = truths[2].unsqueeze(0).unsqueeze(0)
        figure = plt.figure()
        figure.add_subplot(4, 1, 1)
        plt.title('Input Image')
        plt.imshow(input.squeeze().to('cpu').detach().numpy())
        figure.add_subplot(4, 1, 2)
        plt.title('Ground Truth')
        plt.imshow(comparison.squeeze().to('cpu').detach().numpy())
        figure.add_subplot(4, 1, 3)
        plt.title('Saliency Map')
        plt.imshow(saliency_map.squeeze().to('cpu').detach().numpy())
        figure.add_subplot(4, 1, 4)
        plt.title('Predictions')
        plt.imshow(contour_image)
        plt.title(f'Number of Fasteners in Image: {contour_number}')


        plt.show()

end_time = time.time()
duration = end_time - start_time
print(f"Average FPS = {frames_to_render / duration}")
