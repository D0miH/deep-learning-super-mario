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
from gym.wrappers import Monitor
action_count = 7 # SIMPLE_MOVEMENT
import cv2

import neuroevolution
import model

def convert_image(input_image):
    image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    return cv2.resize(image, (32,32))

def run_agent(agent, rendering=False, monitoring=False, print_reward=False):

    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env.seed(42)

    if monitoring:
        env = Monitor(env, './video', force=True)
    agent.eval()

    state = env.reset()
    if rendering:
        env.render()

    #Conv2d without flatten()
    state = convert_image(state)#.flatten()
    state_list = [state, state, state, state]
    position = -1

    global_reward=0
    s=0
    for _ in range(10000):
        #Conv2d input
        input = torch.from_numpy(np.array(state_list)).type('torch.FloatTensor')\
            .unsqueeze(0)

        #Linear input
        #input = torch.tensor(state_list).type("torch.FloatTensor").view(1,-1)

        output_probabilities = agent(input).detach().numpy()[0]
        action = np.random.choice(range(action_count), 1, \
            p=output_probabilities).item()
        new_state, reward, done, info = env.step(action)
        global_reward += reward

        s=s+1
        if rendering:
            env.render()

        state_list.pop()
        #Conv2d without flatten()
        state_list.append(convert_image(new_state))#.flatten())

        # if mario gets stuck, it gets punished and the loop gets broken
        if position == info["x_pos"]:
            stuck += 1
            if stuck == 100:
                global_reward -= 100
                break
        else:
            stuck = 0

        position = info["x_pos"]
        #env.render()
        #Mario died
        if info["life"] < 2:
            break
    if print_reward:
        print(global_reward)

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
