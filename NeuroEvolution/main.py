import mariogame
import neuroevolution
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import copy
import sys
import os
import pickle
DIRNAME = os.path.dirname(__file__)

train_mario = True
load_elite_mario = True

if train_mario:
    elite_agent = None
    if load_elite_mario:
        with open(os.path.join(DIRNAME, "Output/fittestMario.txt"), 'rb') as input:
            elite_agent = pickle.load(input)

    neuroevolution.main(elite_agent)
else:
    with open(os.path.join(DIRNAME, "Output/fittestMario.txt"), 'rb') as input:
        fittestMario = pickle.load(input)
        mariogame.run_agent(fittestMario, True, True, True)
