from itertools import count

import gym
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.distributions import Categorical
from torch.functional import F
from torch import nn


class PolicyNet(nn.Module):
    episode_durations = []  # the history of durations of each episode for plotting

    state_history = []  # the history of states for one episode
    action_history = []  # the history of actions for one episode
    reward_history = []  # the history of reward for one episode
    steps = 0  # the steps taken in one episode

    def __init__(self, gamma=0.99, optimizer=torch.optim.Adam, learning_rate=0.01):
        super(PolicyNet, self).__init__()

        # create the environment
        self.env = gym.make('CartPole-v0')
        # set the max steps to 500
        self.env._max_expisode_steps = 500
        # set a seed for reproducibility
        self.env.seed(1)

        # set the discount factor and the batch_size
        self.gamma = gamma

        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 36)
        self.fc3 = nn.Linear(36, 2)  # one value for left or right

        self.optimizer = optimizer(self.parameters(), lr=learning_rate)

    def __del__(self):
        # need to close the environment otherwise it crashes on osx
        self.env.close()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)  # get the probability for left or right
        return x

    def play_episode(self, render=False):
        # reset the environment after playing one episode
        state = self.env.reset()
        state = torch.from_numpy(state).float()

        # while we are not done keep iterating
        for timeStep in count():
            # choose the next action
            action_probs = self.forward(state)
            action_prob_dist = Categorical(action_probs)
            action = action_prob_dist.sample().item()

            next_state, reward, done, _ = self.env.step(action)

            if render:
                self.env.render(mode='rgb_array')

            self.state_history.append(state)
            self.action_history.append(action)
            self.reward_history.append(reward)

            state = next_state
            state = torch.from_numpy(state).float()

            self.steps += 1

            if done:
                self.episode_durations.append(timeStep + 1)
                break

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

    # Plot duration curve:
    # From http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.FloatTensor(self.episode_durations)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
