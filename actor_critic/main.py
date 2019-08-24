from itertools import count
import torch
import numpy as np
import matplotlib.pyplot as plt

import gym_super_mario_bros
from gym.wrappers import Monitor
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from wrappers import wrapper
from actor_critic.agent import TwoNetAgent, TwoHeadAgent

WORLD = 1
STAGE = 1
LEVEL_NAME = "SuperMarioBros-{}-{}-v0".format(WORLD, STAGE)
ACTION_SPACE = SIMPLE_MOVEMENT
FRAME_DIM = (84, 84, 4)
FRAME_SKIP = 4
NUM_EPISODES = 20_000
# LEARNING_RATE = 0.00003
ACTOR_LEARNING_RATE = 0.000005
CRITIC_LEARNING_RATE = 0.000007
GAMMA = 0.99
ENTROPY_SCALING = 0.01

RENDER_GAME = True
PLOT_INTERVAL = 50
VIDEO_INTERVAL = 100
CHECKPOINT_INTERVAL = 100
#MODEL_PATH = "./models/actor_critic_two_head_world1-1"
ACTOR_MODEL_PATH = "./models/actor_model_world1-1"
CRITIC_MODEL_PATH = "./models/critic_model_wordl1-1"
LOAD_MODEL = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_environment():
    """Creates the environment, applies some wrappers and returns it."""
    tmp_env = gym_super_mario_bros.make(LEVEL_NAME)
    tmp_env = JoypadSpace(tmp_env, ACTION_SPACE)
    tmp_env = wrapper(tmp_env, FRAME_DIM, FRAME_SKIP)

    return tmp_env


def plot_reward_history(reward_history, mean_reward_history):
    plt.plot(reward_history, "b-", mean_reward_history, "r-")
    plt.ylabel("Rewards")
    plt.xlabel("Episodes")
    plt.show()


def lazy_frame_to_tensor(lazy_frame):
    # pytorch expects the frames as height x width x depth
    return torch.from_numpy(
        np.expand_dims(np.asarray(lazy_frame).astype(np.float64).transpose((2, 1, 0)), axis=0)).float()


def record_one_episode(agent, episode):
    tmp_env = gym_super_mario_bros.make(LEVEL_NAME)
    tmp_env = JoypadSpace(tmp_env, ACTION_SPACE)
    tmp_env = Monitor(tmp_env, './videos/video-episode-{0:05d}'.format(episode), force=True)
    tmp_env = wrapper(tmp_env, FRAME_DIM, FRAME_SKIP)

    state = lazy_frame_to_tensor(tmp_env.reset())

    total_reward = 0
    while True:
        action = agent.get_action(state)

        next_state, reward, done, info = tmp_env.step(action)
        next_state = lazy_frame_to_tensor(next_state)

        if done:
            break

        total_reward += reward

        state = next_state


env = create_environment()
# set all options for reproducability
env.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

agent = TwoNetAgent(frame_dim=FRAME_DIM, action_space_size=env.action_space.n, lr_actor=ACTOR_LEARNING_RATE,
                    lr_critic=CRITIC_LEARNING_RATE, gamma=GAMMA, entropy_scaling=ENTROPY_SCALING,
                    device=DEVICE)
if LOAD_MODEL:
    agent.load_model(actor_model_path=ACTOR_MODEL_PATH, critic_model_path=CRITIC_MODEL_PATH)

record_one_episode(agent, 1)

reward_history = []
mean_reward_history = [0]
for episode in range(1, NUM_EPISODES):

    # reset the environment before a new episode
    state = env.reset()
    state = lazy_frame_to_tensor(state)

    trajectory = []
    total_episode_reward = 0
    total_episode_score = 0
    for step in count(1):
        # get the next action
        action = agent.get_action(state)
        # preform the action
        next_state, reward, done, info = env.step(action)
        next_state = lazy_frame_to_tensor(next_state)

        # add the score to the reward. 1 reward for 100 points
        score_delta = (info["score"] - total_episode_score)
        total_episode_score += score_delta
        reward += (score_delta / 100)

        # add the transition to the trajectory
        trajectory.append([state, action, reward, done])

        # keep track of the total reward
        total_episode_reward += reward

        if done:
            if info["flag_get"]:
                print("Finished Level")
            reward_history.append(total_episode_reward)
            if episode > 200:
                mean_reward_history.append(np.mean(reward_history[-200:]))
            break

        if RENDER_GAME:
            env.render()

        del state
        state = next_state

    # update the model using the trajectory
    actor_loss, critic_loss = agent.update(trajectory)

    print("Episode: {}\t Reward: {:.2f}\t AverageReward: {:.2f}\t Actor Loss: {:.5f}\t Critic Loss: {:.5f}".format(
        episode, total_episode_reward, mean_reward_history[-1], actor_loss, critic_loss))

    if episode % PLOT_INTERVAL == 0:
        plot_reward_history(reward_history, mean_reward_history)
    if episode % VIDEO_INTERVAL == 0:
        record_one_episode(agent, episode)
    if episode % CHECKPOINT_INTERVAL == 0:
        agent.save_model(actor_model_path=ACTOR_MODEL_PATH, critic_model_path=CRITIC_MODEL_PATH)

env.close()
