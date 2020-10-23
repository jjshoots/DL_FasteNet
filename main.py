#!/usr/bin/env python3
import time
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF

from PIL import Image
import matplotlib.pyplot as plt

from ImageLoader import ImageLoader
from FasteNet_Net import FasteNet

# params
DIRECTORY = os.path.dirname(__file__)
DIRECTORY2 = DIRECTORY # 'C:\TEMP'

VERSION_NUMBER = 2
MARK_NUMBER = 1

BATCH_SIZE = 50

THRESHOLD = 0.8

# select device
device = 'cpu'
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
print('USING DEVICE', device)

# holder for images and groundtruths and lowest running loss
images = []
truths = []
lowest_running_loss = 1000

# read in the images
for i in range(3):
    image_path = os.path.join(DIRECTORY, f'img/raw{i+1}.png')
    truth_path = os.path.join(DIRECTORY, f'img/ground{i+1}.png')

    # read images, normalize truth to 1
    image = TF.to_tensor(Image.open(image_path))[0, :, :]
    truth = TF.to_tensor(Image.open(truth_path))[0, :, :]
    truth /= torch.max(truth)
    
    images.append(image)
    truths.append(truth)

# feed data to the ImageLoader and start the datalaoder to generate batch sizes
dataset = ImageLoader(images, truths, (images[0].shape[0], images[0].shape[1]), crop_size=51)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# set this to one to view the output of the dataset
if 0:
    dataiter = iter(dataloader)
    user_input = None
    while user_input != 'Y':
        data, label = dataiter.next()
        
        print(label[0].shape)
        print(data[0].shape)
        plt.imshow(data[0].squeeze())
        plt.show()
        plt.imshow(label[0].squeeze())
        plt.show()

        user_input = input('Key in "Y" to end display, enter to continue...')

# set up net
FasteNet = FasteNet().to(device)

# get the latest pth file and use as weights
WEIGHT_FILE = os.path.join(DIRECTORY2, f'weights/Version{VERSION_NUMBER}/weights{MARK_NUMBER}.pth')

while os.path.isfile(WEIGHT_FILE):
    MARK_NUMBER += 1
    WEIGHT_FILE = os.path.join(DIRECTORY2, f'weights/Version{VERSION_NUMBER}/weights{MARK_NUMBER}.pth')

MARK_NUMBER -= 1
WEIGHT_FILE = os.path.join(DIRECTORY2, f'weights/Version{VERSION_NUMBER}/weights{MARK_NUMBER}.pth')

FasteNet.load_state_dict(torch.load(WEIGHT_FILE))
print(F"Using weights file: {WEIGHT_FILE}")

# set up loss function and optimizer
loss_function = nn.MSELoss()
optimizer = optim.SGD(FasteNet.parameters(), lr=0.001, momentum=0.9)

#  start training
for epoch in range(1000):

    running_loss = 0.0

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


        loss = loss_function(output, labels)

        # backprop
        loss.backward()
        optimizer.step()
        running_loss += loss
            

        if i % 100 == 0:
            # at the moment, no way to evaluate the current state of training, so we just record the current running loss
            lowest_running_loss = (running_loss.item() if (i == 100) else lowest_running_loss)
            
            print(f'Epoch {epoch}; Batch Number {i}; Running Loss {running_loss.item()}; Lowest Running Loss {lowest_running_loss}')
            
            # save the network if the current running loss is lower than the one we have
            if(running_loss.item() < lowest_running_loss) and i > 1:
                # print current status
                
                # save the net
                lowest_running_loss = running_loss.item()
                MARK_NUMBER += 1
                WEIGHT_FILE = os.path.join(DIRECTORY2, f'weights/Version{VERSION_NUMBER}/weights{MARK_NUMBER}.pth')
                torch.save(FasteNet.state_dict(), WEIGHT_FILE)

                # print the weight file that we saved to
                print(F"New lowest point, saving weights to: {WEIGHT_FILE}")

            # reset the running loss for the next n batches
            running_loss = 0.




# FOR INFERENCING
# FOR INFERENCING
# FOR INFERENCING

# set frames to render > 0 to perform inference
frames_to_render = 0
start_time = time.time()
torch.no_grad()
FasteNet.eval()

# set to true for inference
for _ in range(frames_to_render):
    input = images[1].unsqueeze(0).unsqueeze(0).to(device)
    output = FasteNet.forward(input)
    
    torch.cuda.synchronize()

    # set to true to display images
    if 0:
        comparison = truths[1].unsqueeze(0).unsqueeze(0)
        plt.imshow(input.squeeze().to('cpu').detach().numpy())
        plt.show()
        plt.imshow(comparison.squeeze().to('cpu').detach().numpy())
        plt.show()
        plt.imshow(output.squeeze().to('cpu').detach().numpy())
        plt.show()

end_time = time.time()
duration = end_time - start_time
print(f"Average FPS = {frames_to_render / duration}")
