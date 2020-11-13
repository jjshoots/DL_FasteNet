#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as func

# weights file is version 4

class FasteNet_HyperLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm1 = nn.BatchNorm2d(num_features=1)
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.layer2 = nn.MaxPool2d(kernel_size=2)
        self.batch_norm2 = nn.BatchNorm2d(num_features=32)
        self.layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.layer4 = nn.MaxPool2d(kernel_size=2)
        self.batch_norm3 = nn.BatchNorm2d(num_features=64)
        self.layer5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.layer6 = nn.MaxPool2d(kernel_size=2)
        self.batch_norm4 = nn.BatchNorm2d(num_features=128)
        self.layer7 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=1)
        self.output_layer = nn.Sigmoid()
        self.threshold_layer = nn.Threshold(0.8, value=0)

    def forward(self, input):
        x = self.batch_norm1(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = func.relu(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.batch_norm3(x)
        x = func.relu(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.batch_norm4(x)
        x = func.relu(x)
        x = self.layer7(x)
        x = self.output_layer(x)

        if not self.training:
            x = self.threshold_layer(x)
            x = torch.ceil(x)
            
        return x