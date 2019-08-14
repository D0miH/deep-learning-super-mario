import sys
from itertools import count

import cv2
import gym_super_mario_bros
from gym.wrappers import Monitor
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

import matplotlib.pyplot as plt
import numpy as np
import torch

from policy_gradient.agent import Agent
from wrappers import wrapper

# env settings
WORLD = 1
STAGE = 1
LEVEL_NAME = "SuperMarioBros-{}-{}-v0".format(WORLD, STAGE)
FRAME_DIM = (120, 128, 4)  # original image size is 240x256
ACTION_SPACE = COMPLEX_MOVEMENT
RENDER_GAME = True

MODEL_PATH = ""  # to create a new model set it to ""

# training hyperparameters
TRAIN_MODEL = True
LEARNING_RATE = 0.000005
NUM_EPOCHS = 1_000
GAMMA = 0.99

LOG_INTERVAL = 1
PLOT_INTERVAL = 10
VIDEO_INTERVAL = 50

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


def plot_rewards(reward_list, reward_mean_history):
    plt.plot(reward_list, "b-", reward_mean_history, "r-")
    plt.ylabel("Rewards")
    plt.xlabel("Episodes")
    plt.show()


def record_one_episode(agent):
    tmp_env = gym_super_mario_bros.make(LEVEL_NAME)
    tmp_env = JoypadSpace(tmp_env, ACTION_SPACE)
    tmp_env = Monitor(tmp_env, './video', force=True)
    tmp_env = wrapper(tmp_env, FRAME_DIM)

    state = lazy_frame_to_tensor(tmp_env.reset())

    total_reward = 0
    while True:
        action, _ = agent.select_action_based_on_state(state)

        next_state, reward, done, info = tmp_env.step(action)
        next_state = lazy_frame_to_tensor(next_state)

        if done:
            break

        total_reward += reward

        state = next_state


env = create_environment()

agent = Agent(env.action_space.n, FRAME_DIM, LEARNING_RATE, GAMMA, DEVICE, MODEL_PATH)

if not TRAIN_MODEL:
    record_one_episode(agent)
    sys.exit()

reward_history = []
reward_mean_history = []

step_log_prob_history = []
step_reward_history = []

# save one example warped image for preview
state = env.reset()
cv2.imwrite("exampleWarpedImage.jpg", np.asarray(state))

for episode in range(1, NUM_EPOCHS):
    state, last_reward = lazy_frame_to_tensor(env.reset()), 0

    for step in count():
        # perform an action
        action, log_prob = agent.select_action_based_on_state(state)
        step_log_prob_history.append(log_prob)
        # delete the last state to prevent memory overflow
        del state
        state, reward, done, info = env.step(action)

        if done and reward < 0:
            # if we died the reward will be less than zero
            step_reward_history.append(reward)

            last_reward += reward
            reward_history.append(last_reward)
            if episode >= 100:
                reward_mean_history.append(np.mean(reward_history))
            break

        if done and reward > 0:
            # if we solved the current level give mario the highest possible reward of 15
            step_reward_history.append(15)

            last_reward += 15
            reward_history.append(last_reward)
            if episode >= 100:
                reward_mean_history.append(np.mean(reward_history))
            print("Finished the level")
            break

        state = lazy_frame_to_tensor(state)

        if RENDER_GAME:
            env.render()

        step_reward_history.append(reward)
        last_reward += reward

    loss = agent.update(step_log_prob_history, step_reward_history)

    if episode % LOG_INTERVAL == 0:
        print("Episode {}\tLast Reward: {:.2f}\tAverage reward: {:.2f}\tLoss: {:.2f}".format(episode, last_reward,
                                                                                             reward_mean_history[-1],
                                                                                             loss))
    if episode % PLOT_INTERVAL == 0:
        plot_rewards(reward_history, reward_mean_history)
    if episode % VIDEO_INTERVAL == 0:
        record_one_episode(agent)

    del loss
    del step_reward_history[:]
    del step_reward_history[:]
    step_reward_history = []
    step_log_prob_history = []
