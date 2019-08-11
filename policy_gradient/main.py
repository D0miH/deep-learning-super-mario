from itertools import count

import cv2
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from wrappers import wrapper

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# env settings
LEVEL_NAME = "SuperMarioBros-v0"
FRAME_DIM = (84, 84, 4)  # original image size is 240x256
ACTION_SPACE = COMPLEX_MOVEMENT
RENDER_GAME = True

# training hyperparameters
LEARNING_RATE = 0.0001
NUM_EPOCHS = 1000
GAMMA = 0.99

LOG_INTERVAL = 1
PLOT_INTERVAL = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_environment():
    """Creates the environment, applies some wrappers and returns it."""
    tmp_env = gym_super_mario_bros.make(LEVEL_NAME)
    tmp_env = JoypadSpace(tmp_env, ACTION_SPACE)
    tmp_env = wrapper(tmp_env, FRAME_DIM)

    return tmp_env


def select_action_based_on_state(state, policy_net):
    """Returns the sampled action and the log of the probability density."""
    # get the probability distribution of the prediction
    pred_action_probs = policy_net.forward(state)
    pred_action_dist = Categorical(pred_action_probs)

    # sample an action from the probability distribution
    action = pred_action_dist.sample()
    return action.item(), pred_action_dist.log_prob(action)


def lazyframe_to_tensor(lazy_frame):
    # pytorch expects the frames as height x width x depth
    return torch.from_numpy(
        np.expand_dims(np.asarray(lazy_frame).astype(np.float64).transpose((2, 1, 0)), axis=0)).float().to(DEVICE)


def finish_episode(optimizer, log_prob_history, rewards):
    cumulative_reward = 0
    policy_loss = []
    discounted_rewards = []
    for r in reversed(rewards):
        cumulative_reward = r + GAMMA * cumulative_reward
        discounted_rewards.insert(0, cumulative_reward)

    # normalize the returns
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std() + np.finfo(
        np.float32).eps.item()

    for log_prob, cumulative_reward in zip(log_prob_history, discounted_rewards):
        policy_loss.append(-log_prob * cumulative_reward)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum().to(DEVICE)
    policy_loss.backward()
    optimizer.step()
    del policy_loss


def plot_rewards(reward_list, reward_mean_history):
    plt.plot(reward_list, "b-", reward_mean_history, "r-")
    plt.ylabel("Rewards")
    plt.xlabel("Episodes")
    plt.show()


class Policy(nn.Module):
    def __init__(self, num_actions):
        super(Policy, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=FRAME_DIM[2], out_channels=32, kernel_size=3, stride=2)
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.conv3_bn = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(in_features=87616, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = F.elu(self.conv1_bn(out))

        out = self.conv2(out)
        out = F.elu(self.conv2_bn(out))

        out = self.conv3(out)
        out = F.elu(self.conv3_bn(out))
        out = out.view(out.size()[0], -1)

        out = F.elu(self.fc1(out))
        out = F.softmax(self.fc2(out), dim=-1)

        return out


env = create_environment()
policy = Policy(env.action_space.n).to(DEVICE)
optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

reward_history = []
reward_mean_history = []

step_log_prob_history = []
step_reward_history = []

# save one example warped image for preview
state = env.reset()
cv2.imwrite("exampleWarpedImage.jpg", np.asarray(state))

for episode in range(NUM_EPOCHS):
    state, last_reward = lazyframe_to_tensor(env.reset()), 0

    for step in count():
        # perform an action
        action, log_prob = select_action_based_on_state(state, policy)
        step_log_prob_history.append(log_prob)
        # delete the last state to prevent memory overflow
        del state
        state, reward, done, info = env.step(action)

        if done or info["life"] < 2:
            # the environment is doing some strange things here. We have to ensure that the last reward is negative.
            if reward < 0:
                step_reward_history.append(reward)

            last_reward += reward
            reward_history.append(last_reward)
            reward_mean_history.append(np.mean(reward_history))
            break

        state = lazyframe_to_tensor(state)

        if RENDER_GAME:
            env.render()

        step_reward_history.append(reward)
        last_reward += reward

    if episode % LOG_INTERVAL == 0:
        print("Episode {}\tLast Reward: {:.2f}\tAverage reward: {:.2f}".format(episode, last_reward,
                                                                               reward_mean_history[-1]))

    if episode % PLOT_INTERVAL == 0:
        plot_rewards(reward_history, reward_mean_history)

    finish_episode(optimizer, step_log_prob_history, step_reward_history)
    del step_reward_history[:]
    del step_reward_history[:]
    step_reward_history = []
    step_log_prob_history = []
