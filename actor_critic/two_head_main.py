import sys
from itertools import count

import cv2
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
WORLD = 1
STAGE = 1
LEVEL_NAME = "SuperMarioBros-{}-{}-v0".format(WORLD, STAGE)
FRAME_DIM = (120, 128, 4)  # original image size is 240x256
ACTION_SPACE = SIMPLE_MOVEMENT
RENDER_GAME = True

MODEL_PATH = ""  # set it to "" if you don't want to load a model

TRAIN_MODEL = True
# training hyperparameters
LEARNING_RATE = 0.00003
NUM_EPOCHS = 1000
GAMMA = 0.99  # the discount factor
BETA = 0.001  # scaling factor of the entropy
ZETA = 1  # the scaling of the value loss

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

agent = TwoHeadAgent(env.action_space.n, FRAME_DIM, GAMMA, BETA, ZETA, LEARNING_RATE, DEVICE, MODEL_PATH)

if not TRAIN_MODEL:
    record_one_episode(agent)
    sys.exit()

reward_history = []
reward_mean_history = [0]

# save one example warped image for preview
state = env.reset()
cv2.imwrite("exampleWarpedImage.jpg", np.asarray(state))

for episode in range(1, NUM_EPOCHS):
    state, last_reward = lazy_frame_to_tensor(env.reset()), 0

    trajectory = []
    for step in count():
        # perform an action
        action = agent.get_action(state)
        # delete the last state to prevent memory overflow
        next_state, reward, done, info = env.step(action)
        next_state = lazy_frame_to_tensor(next_state)

        if done and reward < 0:
            # if we died the reward will be less than zero
            trajectory.append([state, action, reward, next_state, done])

            last_reward += reward
            reward_history.append(last_reward)
            if episode >= 100:
                reward_mean_history.append(np.mean(reward_history))
            break

        if done and reward > 0:
            # if we solved the current level give mario the highest possible reward of 15
            trajectory.append([state, action, 15, next_state, done])

            last_reward += 15
            reward_history.append(last_reward)
            if episode >= 100:
                reward_mean_history.append(np.mean(reward_history))
            print("Finished the level")
            break

        if RENDER_GAME:
            env.render()

        trajectory.append([state, action, reward, next_state, done])
        last_reward += reward

        del state
        state = next_state

    del state
    critic_loss, actor_loss = agent.update(trajectory)

    if episode % LOG_INTERVAL == 0:
        print("Episode {}\tLast Reward: {:.2f}\tAverage reward: {:.2f}\tCritic Loss: {:.2f}\tActor Loss: {:.2f}".format(
            episode, last_reward,
            reward_mean_history[-1],
            critic_loss, actor_loss))
    if episode % PLOT_INTERVAL == 0:
        plot_rewards(reward_history, reward_mean_history)
    if episode % VIDEO_INTERVAL == 0:
        record_one_episode(agent)

    del critic_loss, actor_loss
    del trajectory[:]
