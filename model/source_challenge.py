"""
EECS 445 - Introduction to Machine Learning
Winter 2022 - Project 2
Source CNN Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.source import Source
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config


class Source(nn.Module):
    def __init__(self):
        super().__init__()

        ## TODO: define each layer
        padding_size = 2
        self.conv1 = nn.Conv2d(3, 16, (5,5), stride = (2,2), padding = padding_size)
        self.pool = nn.MaxPool2d((2, 2), stride = (2,2))
        self.conv2 = nn.Conv2d(16, 64, (5,5), stride = (2,2), padding = padding_size)
        self.conv3 = nn.Conv2d(64, 8, (5,5), stride = (2,2), padding = padding_size)
        self.fc1 = nn.Linear(32, 8)
        ##

        self.init_weights()

    def init_weights(self):
        torch.manual_seed(42)
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        ## TODO: initialize the parameters for [self.fc1]
        nn.init.normal_(self.fc1.weight, 0.0, 1/32)
        nn.init.constant_(self.fc1.bias, 0.0)
        ##

    def forward(self, x):
        """ You may optionally use the x.shape variables below to resize/view the size of
            the input matrix at different points of the forward pass
        """
        N, C, H, W = x.shape

        ## TODO: forward pass
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.fc1(torch.flatten(x, 1))
        ##

        return x
