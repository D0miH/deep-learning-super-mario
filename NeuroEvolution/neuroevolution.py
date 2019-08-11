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

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import neuroevolution
import model
import mariogame

def return_children(agents, sorted_parent_indexes, elite_count):
    """Returning [N-elite_count] mutated agents from sorted_parent_indexes and
        keeping the best [elite_count] agents unchanged
    """

    children_agents = []

    for i in range(len(agents)-elite_count):
        selected_agent_index = sorted_parent_indexes[np.random.randint(len(sorted_parent_indexes))]
        children_agents.append(mutate(agents[selected_agent_index]))

    elite_children = []
    for i in range(elite_count):
        elite_children.append(agents[sorted_parent_indexes[i]])

    children_agents.extend(elite_children)

    return children_agents

def mutate(agent):
    ''' Simple method to add gaussian noise to the agents
    '''

    child_agent = copy.deepcopy(agent)

    mutation_power = 0.02 #hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf

    for param in child_agent.parameters():
        if(len(param.shape)==4): #weights of Conv2D
            for i0 in range(param.shape[0]):
                for i1 in range(param.shape[1]):
                    for i2 in range(param.shape[2]):
                        for i3 in range(param.shape[3]):

                            param[i0][i1][i2][i3]+= mutation_power * np.random.randn()

        elif(len(param.shape)==2): #weights of linear layer
            for i0 in range(param.shape[0]):
                for i1 in range(param.shape[1]):

                    param[i0][i1]+= mutation_power * np.random.randn()

        elif(len(param.shape)==1): #biases of linear layer or conv layer
            for i0 in range(param.shape[0]):

                param[i0]+=mutation_power * np.random.randn()

    return child_agent

torch.set_grad_enabled(False)
print(DIRNAME)
num_agents = 10
elite_count = 1
top_limit_count = 2

generation_count = 1000

agents = mariogame.return_random_agents(num_agents)

for generation in range(generation_count):
    rewards = mariogame.run_agents_n_times(agents, 3)

    sorted_parent_indexes = np.argsort(rewards)[::-1][:top_limit_count]

    top_rewards = []
    for best_parent in sorted_parent_indexes:
        top_rewards.append(rewards[best_parent])

    print("## Generation {} ##".format(generation))
    print("Mean reward: {}  |   Mean of top 5: {}".format(np.mean(rewards),\
        np.mean(top_rewards[:5])))
    print("Top agents: {}   |   Reward: {}".format(sorted_parent_indexes, \
        top_rewards))
    print("################\n\n")
    sys.stdout.flush()

    with open(os.path.join(DIRNAME, "/Output/fittestMario.txt"), 'wb') as output:
        pickle.dump(agents[sorted_parent_indexes[0]], output, \
            pickle.HIGHEST_PROTOCOL)

    children_agents = return_children(agents, sorted_parent_indexes, elite_count)

    agents = children_agents
