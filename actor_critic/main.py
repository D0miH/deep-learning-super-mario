from collections import namedtuple
from itertools import count

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
FRAME_DIM = (120, 132, 4)  # original image size is 240x256
ACTION_SPACE = RIGHT_ONLY
RENDER_GAME = True

# training hyperparameters
LEARNING_RATE = 0.03
NUM_EPOCHS = 1000
GAMMA = 0.99
MAX_STEPS_PER_EPOCH = 500

LOG_INTERVAL = 1
PLOT_INTERVAL = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_environment():
    """Creates the environment, applies some wrappers and returns it."""
    tmp_env = gym_super_mario_bros.make(LEVEL_NAME)
    tmp_env = JoypadSpace(tmp_env, ACTION_SPACE)
    tmp_env = wrapper(tmp_env, FRAME_DIM)

    return tmp_env


def select_action_based_on_state(given_state, given_actor_critic_net):
    """Returns the sampled action and the log of the probability density."""

    pred_action_dist, pred_future_reward = given_actor_critic_net.forward(given_state)

    # sample an action from the probability distribution
    pred_action_dist = Categorical(pred_action_dist)
    sampled_action = pred_action_dist.sample()
    return sampled_action.item(), SavedAction(pred_action_dist.log_prob(sampled_action), pred_future_reward)


def lazy_frame_to_tensor(lazy_frame):
    # pytorch expects the frames as height x width x depth
    return torch.from_numpy(
        np.expand_dims(np.asarray(lazy_frame).astype(np.float64).transpose((2, 1, 0)), axis=0)).float().to(DEVICE)


def finish_episode(ac_net_optimizer, given_action_history, rewards):
    cumulative_reward = 0

    policy_losses = []  # to save the actors policy losses
    value_losses = []  # to save the critics value loss

    discounted_true_rewards = []
    for r in reversed(rewards):
        cumulative_reward = r + GAMMA * cumulative_reward
        discounted_true_rewards.insert(0, cumulative_reward)

    # normalize the discounted rewards
    discounted_rewards = torch.tensor(discounted_true_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std() + np.finfo(
        np.float32).eps.item()

    # calculate the loss for the actor and the critic
    for (log_prob, value), cumulative_reward in zip(given_action_history, discounted_rewards):
        # calculate the advantage
        advantage = cumulative_reward - value.item()

        # calculate the actors loss
        policy_losses.append(-log_prob * advantage)

        # calculate the critics loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([[cumulative_reward]]).to(DEVICE)))

    # perform backprop
    ac_net_optimizer.zero_grad()
    loss = (torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()).to(DEVICE)
    loss.backward()
    ac_net_optimizer.step()

    del loss


def plot_rewards(reward_list, given_reward_mean_history):
    plt.plot(reward_list, "b-", given_reward_mean_history, "r-")
    plt.ylabel("Rewards")
    plt.xlabel("Episodes")
    plt.legend(["Reward per episode", "Average reward"])
    plt.show()


SavedAction = namedtuple("SavedAction", ["log_prob", "future_reward"])


class ActorCriticNet(nn.Module):
    """
    Neural net to select the next action to take.
    The output is a probability distribution over all possible actions.
    """

    def __init__(self, num_actions):
        super(ActorCriticNet, self).__init__()

        # create a convolution net
        self.conv1 = nn.Conv2d(in_channels=FRAME_DIM[2], out_channels=32, kernel_size=8, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(in_features=89600, out_features=512)

        # create one output for the actor
        self.head_actor = nn.Linear(in_features=512, out_features=num_actions)

        # create one output for the critic
        self.head_critic = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        out = F.elu(self.conv1(x))

        out = F.elu(self.conv2(out))

        out = F.elu(self.conv3(out))

        out = out.view(out.size()[0], -1)

        out = F.elu(self.fc1(out))

        # get the probability dist of the actor
        actor_out = F.softmax(self.head_actor(out), dim=-1)

        # get the predicted future reward of the critic
        critic_out = self.head_critic(out)

        return actor_out, critic_out


env = create_environment()

ac_net = ActorCriticNet(env.action_space.n).to(DEVICE)
optimizer = optim.Adam(ac_net.parameters(), lr=LEARNING_RATE)

reward_history = []
reward_mean_history = []

action_history = []
step_reward_history = []
for episode in range(1, NUM_EPOCHS):
    state, last_reward = lazy_frame_to_tensor(env.reset()), 0

    for step in count(1):
        # perform an action
        action, saved_action = select_action_based_on_state(state, ac_net)
        action_history.append(saved_action)
        # delete the last state to prevent memory overflow
        del state
        state, reward, done, info = env.step(action)

        if done or info["life"] < 2 or step >= MAX_STEPS_PER_EPOCH:
            reward_history.append(last_reward)
            reward_mean_history.append(np.mean(reward_history))
            break

        state = lazy_frame_to_tensor(state)

        if RENDER_GAME:
            env.render()

        step_reward_history.append(reward)
        last_reward += reward

    if episode % LOG_INTERVAL == 0:
        print("Episode {}\tLast Reward: {:.2f}\tAverage reward: {:.2f}".format(episode, last_reward,
                                                                               reward_mean_history[-1]))

    if episode % PLOT_INTERVAL == 0:
        plot_rewards(reward_history, reward_mean_history)

    finish_episode(optimizer, action_history, step_reward_history)
    del step_reward_history[:]
    del action_history[:]
    step_reward_history = []
    action_history = []
