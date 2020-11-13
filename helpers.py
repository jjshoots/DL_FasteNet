#!/usr/bin/env python3
from sys import version
import time
import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF

from PIL import Image
import matplotlib.pyplot as plt
import cv2


class helpers:
    def __init__(self, mark_number, version_number, weights_location):
        # training checkpoint variables
        self.running_loss = 0
        self.lowest_running_loss = 0

        # weight file variables
        self.weights_location = weights_location
        self.mark_number = mark_number
        self.version_number = version_number
        self.weights_file = os.path.join(self.weights_location, f'weights/Version{self.version_number}/weights{self.mark_number}.pth')

        # home directory
        self.directory = os.path.dirname(__file__)



    # simple call to reset internal running loss
    def reset_running_loss(self):
        self.running_loss = 0.



    # call inside training loop
    # helps to display training progress and save any improvement
    def training_checkpoint(self, loss, iterations, epoch):
        self.running_loss += loss

        if iterations % 100 == 0:
            # at the moment, no way to evaluate the current state of training, so we just record the current running loss
            self.lowest_running_loss = (self.running_loss.item() if (iterations == 100) else self.lowest_running_loss)
            
            print(f'Epoch {epoch}; Batch Number {iterations}; Running Loss {self.running_loss}; Lowest Running Loss {self.lowest_running_loss}')

            # save the network if the current running loss is lower than the one we have
            if(self.running_loss.item() < self.lowest_running_loss) and iterations > 1:
                # save the net
                self.lowest_running_loss = self.running_loss.item()
                self.mark_number += 1

                # regenerate the weights_file path
                self.weights_file = os.path.join(self.weights_location, f'weights/Version{self.version_number}/weights{self.mark_number}.pth')

                # reset the running loss for the next n batches
                self.running_loss = 0.

                # print the weight file that we should save to
                print(F"New lowest point, saving weights to: {self.weights_file}")
                return self.weights_file

            # reset the running loss for the next n batches
            self.running_loss = 0.
        
        # return -1 if we are not returning the weights file
        return -1



    # retrieves the latest weight file based on mark and version number
    # weight location is location where all weights of all versions are stored
    # version number for new networks, mark number for training
    def get_latest_weight_file(self):

        # while the file exists, try to look for a file one version later
        while os.path.isfile(self.weights_file):
            self.mark_number += 1
            self.weights_file = os.path.join(self.weights_location, f'weights/Version{self.version_number}/weights{self.mark_number}.pth')

        # once the file version doesn't exist, decrement by one and use that file
        self.mark_number -= 1
        self.weights_file = os.path.join(self.weights_location, f'weights/Version{self.version_number}/weights{self.mark_number}.pth')

        # if there's no files, ignore
        if self.mark_number > 0:
            print(F"Using weights file: {self.weights_file}")
            return self.weights_file
        else:
            print(F"No weights file found, generating new one during training.")
            return -1



#################################################################################################
#################################################################################################
# STATIC FUNCTIONS
#################################################################################################
#################################################################################################

    @staticmethod
    def get_device():
        # select device
        device = 'cpu'
        if(torch.cuda.is_available()):
            device = torch.device('cuda:0')
        print('USING DEVICE', device)

        return device

    # enables peeking of the dataset
    @staticmethod
    def peek_dataset(dataloader):
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

        exit()


    # converts saliency map to pseudo segmentation
    # expects input of dim 2
    # fastener_area_threshold is minimum area of output object BEFORE scaling to input size
    @staticmethod
    def saliency_to_contour(input, original_image, fastener_area_threshold, input_output_ratio):
        # find contours in the image
        threshold = input.detach().cpu().squeeze().numpy().astype(np.uint8)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # filter contours below certain area
        filtered_contours = []

        for contour in contours:
            if cv2.contourArea(contour) > fastener_area_threshold:
                contour *= input_output_ratio
                filtered_contours.append(contour)

        # draw contours
        contour_image = original_image.squeeze().to('cpu').detach().numpy()
        cv2.drawContours(contour_image, filtered_contours, -1, 1, 1)

        # return drawn image
        return contour_image, len(filtered_contours)

