#!/usr/bin/env python3
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
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from helpers import helpers
from ImageLoader import ImageLoader
from FasteNet_Net_v2 import FasteNet_v2

from FasteNet_Vanilla_Net import FasteNet_Vanilla
from FasteNet_Large_Net import FasteNet_Large

# params
DIRECTORY = 'C:\AI\DATA' # os.path.dirname(__file__)
DIRECTORY2 = 'C:\AI\WEIGHTS'
SHUTDOWN_AFTER_TRAINING = False

VERSION_NUMBER = 6
MARK_NUMBER = 1

BATCH_SIZE = 100
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

        if os.path.isfile(image_path): 

            # read images
            image = TF.to_tensor(cv2.imread(image_path))[0]
            truth = TF.to_tensor(cv2.imread(truth_path))[0]
            
            # normalize inputs, 1e-6 for stability as some images don't have truth masks (no fasteners)
            image /= torch.max(image + 1e-6)
            truth /= torch.max(truth + 1e-6)

            images.append(image)
            truths.append(truth)

    print(f'Attempted to load images {from_image} to {to_image}, actually loaded {len(images)} images.')


    # feed data to the ImageLoader and start the dataloader to generate batches
    dataset = ImageLoader(images, truths, (images[0].shape[0], images[0].shape[1]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    return dataloader

# uncomment this to one to view the output of the dataset
# helpers.peek_dataset(dataloader=dataloader)

# set up net
FasteNet = FasteNet_Vanilla().to(device)

# get latest weight file
weights_file = helpers.get_latest_weight_file()
if weights_file != -1:
    FasteNet.load_state_dict(torch.load(weights_file))

# set up loss function and optimizer and load in data
loss_function = nn.MSELoss()
optimizer = optim.SGD(FasteNet.parameters(), lr=1e-6, weight_decay=1e-2)
optimizer = optim.Adam(FasteNet.parameters(), )
# dataloader = generate_dataloader(0)

# get network param number and gflops
# image_path = os.path.join(DIRECTORY, f'Dataset/image/image_{1}.png')
# image = TF.to_tensor(cv2.imread(image_path))[0].unsqueeze(0).unsqueeze(0)[..., :1600].to(device)
# image /= torch.max(image + 1e-6)
# helpers.network_stats(FasteNet, image)

#  start training
for epoch in range(0):
    dataloader = generate_dataloader(epoch % NUMBER_OF_CYCLES)
    helpers.reset_running_loss()

    for i, data in enumerate(dataloader):
        # pull data out from dataloader
        data, labels = data[0].to(device), data[1].to(device)

        # zero the graph gradient
        FasteNet.zero_grad()

        # get the output and calculate the loss
        output = FasteNet.forward(data)

        if False:
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

if SHUTDOWN_AFTER_TRAINING:
    os.system("shutdown /s /t 30")
    exit()

# exit()

# HARD NEGATIVE MINING
# HARD NEGATIVE MINING
# HARD NEGATIVE MINING

# params
torch.no_grad()
FasteNet.eval()
negative_mined_number = 0

total_true_positive = 1e-6
total_false_positive = 1e-6
total_false_negative = 1e-6

# set to true for inference
for index in range(700, 997):
    image_path = os.path.join(DIRECTORY, f'Dataset/image/image_{index}.png')
    truth_path = os.path.join(DIRECTORY, f'Dataset/label/label_{index}.png')

    # read images
    image = TF.to_tensor(cv2.imread(image_path))[0]
    truth = TF.to_tensor(cv2.imread(truth_path))[0]
    
    # normalize inputs, 1e-6 for stability as some images don't have truth masks (no fasteners)
    image /= torch.max(image + 1e-6)
    truth /= torch.max(truth + 1e-6)


    input = image.unsqueeze(0).unsqueeze(0).to(device)[..., :1600]
    saliency_map = FasteNet.forward(input)
    torch.cuda.synchronize()

    # calculate true positive number
    _, true_positive_number = helpers.saliency_to_contour(input=saliency_map.to('cpu') * F.interpolate(truth.unsqueeze(0).unsqueeze(0)[..., :1600] * 255, size=[saliency_map.shape[-2], saliency_map.shape[-1]]), original_image=None, fastener_area_threshold=1, input_output_ratio=8)

    # draw contours on original image and prediction image
    _, contour_number = helpers.saliency_to_contour(input=saliency_map, original_image=None, fastener_area_threshold=1, input_output_ratio=8)
    _, ground_number = helpers.saliency_to_contour(input=truth.unsqueeze(0).unsqueeze(0)[..., :1600] * 255, original_image=None, fastener_area_threshold=1, input_output_ratio=1)

    false_positive_number = contour_number - true_positive_number
    false_negative_number = ground_number - true_positive_number

    if(false_positive_number < 0):
        print(f'Image {index} has more true positives than contour, likely because area threshold is not large enough.')

    total_true_positive += true_positive_number
    total_false_positive += false_positive_number
    total_false_negative += false_negative_number

    if abs(ground_number - contour_number) > 2:
        continue
        # read images
        image = cv2.imread(image_path)
        truth = cv2.imread(truth_path)

        # hard negative image path
        save_image_path = os.path.join(DIRECTORY, f'Dataset/hard_negative/image/image_{index}.png')
        save_truth_path = os.path.join(DIRECTORY, f'Dataset/hard_negative/label/label_{index}.png')

        # save the images
        cv2.imwrite(save_image_path, image)
        cv2.imwrite(save_truth_path, truth)

        negative_mined_number += 1

print(f'Total images mined: {negative_mined_number}')
print(f'Precision: {total_true_positive / (total_true_positive + total_false_positive)}')
print(f'Recall: {total_true_positive / (total_true_positive + total_false_negative)}')

exit()

# FOR INFERENCING
# FOR INFERENCING
# FOR INFERENCING

# set frames to render > 0 to perform inference
torch.no_grad()
FasteNet.eval()
frames_to_render = 100
start_time = time.time()


# set to true for inference
for _ in range(frames_to_render):
    index = nprand.randint(0, 996)

    image_path = os.path.join(DIRECTORY, f'Dataset/image/image_{index}.png')
    truth_path = os.path.join(DIRECTORY, f'Dataset/label/label_{index}.png')

    # read images
    image = TF.to_tensor(cv2.imread(image_path))[0]
    truth = TF.to_tensor(cv2.imread(truth_path))[0]
    
    # normalize inputs, 1e-6 for stability as some images don't have truth masks (no fasteners)
    image /= torch.max(image + 1e-6)
    truth /= torch.max(truth + 1e-6)

    input = image.unsqueeze(0).unsqueeze(0).to(device)[..., :1600]
    saliency_map = FasteNet.forward(input)
    torch.cuda.synchronize()

    # draw contours on original image and prediction image
    contour_image, contour_number = helpers.saliency_to_contour(input=saliency_map, original_image=input, fastener_area_threshold=0, input_output_ratio=8)
    ground_image, ground_number = helpers.saliency_to_contour(input=truth.unsqueeze(0).unsqueeze(0)[..., :1600] * 255, original_image=input, fastener_area_threshold=1, input_output_ratio=1)

    # use this however you want to use it
    image_image = np.array(cv2.imread(image_path)[..., 0][:, :1600], dtype=np.float64)
    image_image /= 205
    fused_image = np.transpose(np.array([ground_image, contour_image, image_image]), [1, 2, 0])

    # set to true to display images
    if True:
        figure = plt.figure()

        figure.add_subplot(2, 2, 1)
        plt.title(f'Input Image: Index {index}')
        plt.imshow(input.squeeze().to('cpu').detach().numpy(), cmap='gray')
        figure.add_subplot(2, 2, 2)
        plt.title('Ground Truth')
        plt.imshow(ground_image, cmap='gray')
        plt.title(f'Ground Truth Number of Fasteners in Image: {ground_number}')
        figure.add_subplot(2, 2, 3)
        plt.title('Saliency Map')
        plt.imshow(saliency_map.squeeze().to('cpu').detach().numpy(), cmap='gray')
        figure.add_subplot(2, 2, 4)
        plt.title('Predictions')
        plt.imshow(fused_image)
        plt.title(f'Predicted Number of Fasteners in Image: {contour_number}')

        plt.show()

end_time = time.time()
duration = end_time - start_time
print(f"Average FPS = {frames_to_render / duration}")

