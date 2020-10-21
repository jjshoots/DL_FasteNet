#!/usr/bin/env python3
import time
import os

import torch
from torch._C import Size
import torch.nn as nn
import torch.nn.functional as func

class FasteNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.layer2 = nn.MaxPool2d(kernel_size=2)
        self.layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.layer4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.layer5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.layer6 = nn.MaxPool2d(kernel_size=2)
        self.layer7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.layer8 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.layer9 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.layer10 = nn.MaxPool2d(kernel_size=2)
        self.layer11 = nn.Conv2d(in_channels=128, out_channels=5, kernel_size=3, padding=1)

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        return x