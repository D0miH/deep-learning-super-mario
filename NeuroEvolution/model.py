import numpy as np
import torch
import matplotlib.pyplot as plt
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import copy

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import neuroevolution
import model
import mariogame


class NeuralMario(nn.Module):
        def __init__(self, action_count):
            super().__init__()
            self.fc = nn.Sequential(
                        nn.Linear(512, 128, bias=True),
                        nn.ReLU(),
                        nn.Linear(128, action_count, bias=True),
                        nn.Softmax(dim=1)
                        )


        def forward(self, inputs):
            x = self.fc(inputs)
            return x

def init_weights(m):

        # nn.Conv2d weights are of shape [16, 1, 3, 3] i.e. # number of filters, 1, stride, stride
        # nn.Conv2d bias is of shape [16] i.e. # number of filters

        # nn.Linear weights are of shape [32, 24336] i.e. # number of input features, number of output features
        # nn.Linear bias is of shape [32] i.e. # number of output features

        if ((type(m) == nn.Linear) | (type(m) == nn.Conv2d)):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.00)
