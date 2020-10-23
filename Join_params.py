#!/usr/bin/env python3
import os
from collections import OrderedDict

import torch
from torch._C import Size
import torch.nn as nn
import torch.nn.functional as func

class newnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold_layer = nn.Threshold(0.5, value=0)
        # self.THRESHOLD = nn.Parameter(nn.Parameter(torch.ones(1).cuda()))

DIRECTORY = os.path.dirname(__file__)
WEIGHT_FILE = os.path.join(DIRECTORY, f'weights/Version1/weights156.pth')
state1 = torch.load(WEIGHT_FILE)

net = newnet()
state2 = net.state_dict()

DIRECTORY = os.path.dirname(__file__)
WEIGHT_FILE = os.path.join(DIRECTORY, f'weights/Version2/weights1.pth')

state = OrderedDict(list(state1.items()) + list(state2.items())) 
torch.save(state, WEIGHT_FILE)



