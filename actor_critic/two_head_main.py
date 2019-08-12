from itertools import count

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from gym.wrappers import Monitor
from nes_py.wrappers import JoypadSpace

from wrappers import wrapper
from actor_critic.two_head_agent import TwoHeadAgent

import matplotlib.pyplot as plt
import numpy as np
import torch

# env settings
LEVEL_NAME = "SuperMarioBros-v2"
FRAME_DIM = (84, 84, 4)  # original image size is 240x256
ACTION_SPACE = SIMPLE_MOVEMENT
RENDER_GAME = True

# training hyperparameters
LEARNING_RATE = 0.00003
NUM_EPOCHS = 1000
GAMMA = 0.99  # the discount factor
ZETA = 1  # the scaling of the value loss

LOG_INTERVAL = 1
PLOT_INTERVAL = 10
VIDEO_INTERVAL = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_environment():
    """Creates the environment, applies some wrappers and returns it."""
    tmp_env = gym_super_mario_bros.make(LEVEL_NAME)
    tmp_env = JoypadSpace(tmp_env, ACTION_SPACE)
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


def record_one_episode(agent):
    env = create_environment()
    env = Monitor(env, './video', force=True)

    state = lazy_frame_to_tensor(env.reset())

    total_reward = 0
    while True:
        action = agent.get_action(state)

        next_state, reward, done, info = env.step(action)
        next_state = lazy_frame_to_tensor(next_state)

        if done:
            break

        total_reward += reward

        state = next_state


env = create_environment()
env.seed(42)
torch.manual_seed(42)

agent = TwoHeadAgent(env.action_space.n, FRAME_DIM, GAMMA, ZETA, LEARNING_RATE, DEVICE)

reward_history = []
reward_mean_history = []
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

        if info["life"] < 2:
            # the environment is doing some strange things here. We have to ensure that the last reward is negative.
            if reward < 0:
                trajectory.append([state, action, reward, next_state, done])

            reward_history.append(episode_reward)
            reward_mean_history.append(np.mean(reward_history))
            break

        trajectory.append([state, action, reward, next_state, done])

        # update the reward of the episode
        episode_reward += reward

        # update the state and delete the previous one
        state = next_state

        if RENDER_GAME:
            env.render()

        if done:
            reward_history.append(episode_reward)
            reward_mean_history.append(np.mean(reward_history))
            break

    # update the agent based on the trajectory
    critic_loss, actor_loss = agent.update(trajectory)
    # delete the trajectory to save memory
    del trajectory

    # log some info
    if episode % LOG_INTERVAL == 0:
        print("Episode {}\tReward: {:.2f}\tAverage reward: {:.2f}\tActor Loss: {:.2f}\tCritic Loss: {:.2f}".format(
            episode, episode_reward,
            reward_mean_history[-1], actor_loss, critic_loss))
    del critic_loss, actor_loss
    if episode % PLOT_INTERVAL == 0:
        plot_rewards(reward_history, reward_mean_history)
        agent.plot_loss()
    if episode % VIDEO_INTERVAL == 0:
        agent.model.eval()
        record_one_episode(agent)
        agent.model.train()

    torch.cuda.empty_cache()
