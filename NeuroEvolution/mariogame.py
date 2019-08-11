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
action_count = 7 # SIMPLE_MOVEMENT
import cv2

import neuroevolution
import model
import mariogame

def convert_image(input_image):
    image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    return cv2.resize(image, (1,128))

def run_agent(agent):

    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env.seed(42)

    agent.eval()

    state = env.reset()
    state = convert_image(state).flatten()
    state_list = [state, state, state, state]

    global_reward=0
    s=0
    while True:
        input = torch.tensor(state_list).type('torch.FloatTensor').view(1,-1)
        output_probabilities = agent(input).detach().numpy()[0]
        action = np.random.choice(range(action_count), 1, p=output_probabilities).item()
        new_state, reward, done, info = env.step(action)
        global_reward += reward

        s=s+1

        state_list.pop(0)
        state_list.append(convert_image(new_state))

        #Mario died
        if info["life"] < 2:
            break

    return global_reward

def return_average_score(agent, runs):
    score = 0.
    for _ in range(runs):
        score += run_agent(agent)
    return score/runs

def run_agents_n_times(agents, runs):
    avg_score = []
    for agent in agents:
        avg_score.append(return_average_score(agent,runs))
    return avg_score


def return_random_agents(num_agents):
    agents = []
    for _ in range(num_agents):

        agent = model.NeuralMario(action_count)

        for param in agent.parameters():
            param.requires_grad = False

        model.init_weights(agent)
        agents.append(agent)

    return agents
