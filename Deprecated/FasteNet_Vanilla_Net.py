#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.modules import module

# weights file is version 6

class FasteNet_Vanilla(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # input normalization
            nn.BatchNorm2d(num_features=1),

            # upscaling module
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            # hidden module 1
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),

            # hidden module 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),

           # hidden module 3
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),

            # output module
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.threshold_layer = nn.Threshold(0.9, value=0)

    def forward(self, input):
        x = self.net(input)
        
        if not self.training:
            x = self.threshold_layer(x)
            x = torch.ceil(x)
        
        return x