from itertools import count

import cv2
from torch import nn
from torch.distributions import Categorical
from torch.functional import F
import torch

import numpy as np

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


class MarioNet(nn.Module):
    state_history = []
    action_history = []
    reward_history = []
    steps = 0

    def __init__(self, gamma=0.99, optimizer=torch.optim.Adam, learning_rate=0.01):
        super(MarioNet, self).__init__()

        # create the environment
        self.env = gym_super_mario_bros.make("SuperMarioBros-v0")
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)

        # set a seed for reproducibility
        self.env.seed(42)
        torch.manual_seed(42)

        # set the discount factor
        self.gamma = gamma

        # create the layers for the nn
        # each greyscale frame is 240(height)x256(width) pixels. We stack 4 images as input.
        # 4 input images, 32 filters (#activation maps), kernel size of 8 and stride of 4
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=12, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=2, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(1456, 512)  # input is calculated by 2*26*28
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.env.action_space.n)
        self.sm5 = nn.Softmax(dim=-1)

        self.optimizer = optimizer(self.parameters(), lr=learning_rate)

    def __del__(self):
        self.env.close()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        # reshape the output for the linear layer
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        out = self.sm5(out)

        return out

    def convert_to_greyscale(self, input_image):
        return cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

    def play_episode(self, render=False):
        # get the first image as greyscale
        img_data = self.env.reset()
        img_data = self.convert_to_greyscale(img_data)
        img_data = torch.from_numpy(img_data).float()

        # state is a list of images
        img_list = [img_data, img_data, img_data, img_data]

        # list to check if mario is moving
        x_pos_list = [i for i in range(5)]

        # while we are not done keep iterating
        for _ in count():
            # choose the next action
            state = torch.stack(img_list).unsqueeze(0)
            action_probs = self.forward(state)
            action_probs_dist = Categorical(action_probs)
            action = action_probs_dist.sample().item()

            # take the action
            next_image, reward, done, info = self.env.step(action)

            # append the current x position to the list
            current_x_pos = info["x_pos"]
            if len(x_pos_list) > 100:
                x_pos_list.pop(0)
            x_pos_list.append(current_x_pos)
            # if mario didn't move end the episode
            if all(x_pos == x_pos_list[0] for x_pos in x_pos_list):
                break

            if done:
                break

            if render:
                self.env.render()

            self.state_history.append(state)
            self.action_history.append(action)
            self.reward_history.append(reward)

            # remove the last picture of the list and insert the new one
            img_list.pop(0)
            next_image = self.convert_to_greyscale(next_image)
            img_list.append(torch.from_numpy(next_image).float())

            self.steps += 1

    def train_on_played_episode(self):
        # discount the rewards
        cumulative_reward = 0
        for i in reversed(range(self.steps)):
            cumulative_reward = cumulative_reward * self.gamma + self.reward_history[i]
            self.reward_history[i] = cumulative_reward

        # Normalize rewards
        reward_mean = np.mean(self.reward_history)
        reward_std = np.std(self.reward_history)
        self.reward_history = (self.reward_history - reward_mean) / reward_std

        # reset the gradients
        self.optimizer.zero_grad()

        # calculate the loss for each step
        for step in range(self.steps):
            # get the state at the step
            state = self.state_history[step]
            # get the action at the step
            action = torch.FloatTensor([self.action_history[step]])
            # get the reward at the step
            reward = self.reward_history[step]

            action_probs = self.forward(state)
            action_prob_dist = Categorical(action_probs)
            # for minimizing the loss we take the negative of the log policy function (this is the policy gradient)
            loss = -action_prob_dist.log_prob(action) * reward
            loss.backward()

        # update the weights
        self.optimizer.step()

        # reset the histories and the number of steps
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.steps = 0
