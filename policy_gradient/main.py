from itertools import count

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from wrappers import wrapper

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# env settings
LEVEL_NAME = "SuperMarioBros-v0"
FRAME_DIM = (120, 132, 4)  # original image size is 240x256
ACTION_SPACE = COMPLEX_MOVEMENT
RENDER_GAME = True

# training hyperparameters
LEARNING_RATE = 0.0001
NUM_EPOCHS = 1000
GAMMA = 0.99
MAX_STEPS_PER_EPOCH = 500

def create_environment():
    """Creates the environment, applies some wrappers and returns it."""
    tmp_env = gym_super_mario_bros.make(LEVEL_NAME)
    tmp_env = JoypadSpace(tmp_env, ACTION_SPACE)
    tmp_env = wrapper(tmp_env, FRAME_DIM)

    return tmp_env


def select_action_based_on_state(state, policy_net):
    """Returns the sampled action."""
    # get the probability distribution of the prediction
    pred_action_probs = policy_net.forward(state)
    pred_action_dist = Categorical(pred_action_probs)

    # sample an action from the probability distribution
    action = pred_action_dist.sample()
    policy_net.saved_log_probs.append(pred_action_dist.log_prob(action))
    return action.item()


def lazyframe_to_tensor(lazy_frame):
    # pytorch expects the frames as height x width x depth
    return torch.from_numpy(np.expand_dims(np.asarray(lazy_frame).transpose((2, 1, 0)), axis=0)).float()


def finish_episode(policy_net, optimizer):
    R = 0
    policy_loss = []
    returns = []
    for r in reversed(policy_net.rewards):
        R = r + GAMMA * R
        returns.insert(0, R)

    # normalize the returns
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / returns.std() + np.finfo(np.float32).eps.item()

    for log_prob, R in zip(policy_net.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy_net.rewards[:]
    del policy_net.saved_log_probs[:]


class Policy(nn.Module):
    def __init__(self, num_actions):
        super(Policy, self).__init__()

        self.saved_log_probs = []
        self.rewards = []

        self.conv1 = nn.Conv2d(in_channels=FRAME_DIM[2], out_channels=32, kernel_size=8, stride=2)
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)
        self.conv3_bn = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(in_features=21504, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)

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
policy = Policy(env.action_space.n)
optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

reward_history = []
for episode in range(NUM_EPOCHS):
    state, last_reward = lazyframe_to_tensor(env.reset()), 0

    for step in range(MAX_STEPS_PER_EPOCH):
        # perform an action
        action = select_action_based_on_state(state, policy)

        state, reward, done, info = env.step(action)
        state = lazyframe_to_tensor(state)

        if RENDER_GAME:
            env.render()

        policy.rewards.append(reward)
        last_reward += reward

        if done or info["life"] < 2:
            reward_history.append(last_reward)
            print("Episode {}\tLast Reward: {:.2f}\tAverage reward: {:.2f}".format(episode, last_reward,
                                                                                   np.mean(reward_history)))
            break

    finish_episode(policy, optimizer)
