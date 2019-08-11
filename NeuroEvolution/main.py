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


train_mario = True

if train_mario:
    neuroevolution.__init__
else:
    with open(os.path.join(DIRNAME, "/Output/fittestMario.txt"), 'rb') as input:
    fittestMario = pickle.load(input)
