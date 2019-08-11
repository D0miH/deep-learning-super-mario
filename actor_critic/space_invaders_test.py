from itertools import count
import gym

from wrappers import wrapper
from actor_critic.agent import Agent

import matplotlib.pyplot as plt
import numpy as np
import torch

# env settings
LEVEL_NAME = "SpaceInvaders-v0"
FRAME_DIM = (84, 84, 4)
RENDER_GAME = True

# training hyperparameters
ACTOR_LEARNING_RATE = 0.0003  # gradient seems to be exploding :/ in particular the entropy loss is exploding maybe we could clip the reward
CRITIC_LEARNING_RATE = 0.0003
NUM_EPOCHS = 1000
GAMMA = 0.99  # the discount factor
BETA = 1  # the scaling of the entropy
ZETA = 1  # the scaling of the value loss

LOG_INTERVAL = 1
PLOT_INTERVAL = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_environment():
    """Creates the environment, applies some wrappers and returns it."""
    tmp_env = gym.make(LEVEL_NAME)
    tmp_env = wrapper(tmp_env, FRAME_DIM)

    return tmp_env


def lazy_frame_to_tensor(lazy_frame):
    # pytorch expects the frames as height x width x depth
    return torch.from_numpy(
        np.expand_dims(np.asarray(lazy_frame).astype(np.float64).transpose((2, 1, 0)), axis=0)).float().to(DEVICE)


def plot_rewards(reward_list, given_reward_mean_history):
    plt.plot(reward_list, "b-", given_reward_mean_history, "r-")
    plt.ylabel("Rewards")
    plt.xlabel("Episodes")
    plt.legend(["Reward per episode", "Average reward"])
    plt.show()


env = create_environment()
env.seed(1)
torch.manual_seed(1)

agent = Agent(env.action_space.n, FRAME_DIM, GAMMA, BETA, ZETA, ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE, DEVICE)

reward_history = []
reward_mean_history = []

total_steps = 0
for episode in range(1, NUM_EPOCHS):
    state = lazy_frame_to_tensor(env.reset())

    episode_reward = 0
    trajectory = []  # in here all the transitions are stored as [[state, action, reward, next_state, done],...]
    for step in count(1):
        # get the next action to perform
        action = agent.get_action(state)

        # perform the action and get the feedback from the environment
        next_state, reward, done, info = env.step(action)
        next_state = lazy_frame_to_tensor(next_state)

        trajectory.append([state, action, reward, next_state, done])

        # update the reward of the episode
        episode_reward += reward

        # update the state and delete the previous one
        state = next_state

        total_steps += 1

        if RENDER_GAME:
            env.render()

        if done:
            reward_history.append(episode_reward)
            reward_mean_history.append(np.mean(reward_history))
            break

    agent.update(trajectory)

    # log some info
    if episode % LOG_INTERVAL == 0:
        print("Episode {}\tReward: {:.2f}\tAverage reward: {:.2f}".format(
            episode, episode_reward,
            reward_mean_history[-1]))

    if episode % PLOT_INTERVAL == 0:
        plot_rewards(reward_history, reward_mean_history)
        agent.plot_loss()

    torch.cuda.empty_cache()
