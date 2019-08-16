import math
from itertools import count
import matplotlib.pyplot as plt

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.functional import F

from wrappers import wrapper

# env settings
WORLD = 1
STAGE = 1
LEVEL_NAME = "SuperMarioBros-{}-{}-v0".format(WORLD, STAGE)
FRAME_DIM = (120, 128, 4)  # original image size is 240x256
ACTION_SPACE = SIMPLE_MOVEMENT
RENDER_GAME = True

MODEL_PATH = "/Users/dominik/Desktop/Projects/deep-learning-super-mario/policy_gradient/models/v0/model_world-1-1"  # to create a new model set it to ""

# training hyperparameters
LEARNING_RATE = 0.000007
GAMMA = 0.99

VISUALIZATION_INTERVAL = 50

FIRST_LAYER_KERNEL_SIZE = 3
FIRST_LAYER_STRIDE = 2
FIRST_LAYER_OUT = 32

SECOND_LAYER_KERNEL_SIZE = 3
SECOND_LAYER_STRIDE = 1
SECOND_LAYER_OUT = 32

THIRD_LAYER_KERNEL_SIZE = 3
THIRD_LAYER_STRIDE = 1
THIRD_LAYER_OUT = 32

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


class Policy(nn.Module):
    def __init__(self, num_actions, frame_dim):
        super(Policy, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=frame_dim[2], out_channels=FIRST_LAYER_OUT,
                               kernel_size=FIRST_LAYER_KERNEL_SIZE, stride=FIRST_LAYER_STRIDE)
        self.conv1_bn = nn.BatchNorm2d(FIRST_LAYER_OUT)

        self.conv2 = nn.Conv2d(in_channels=FIRST_LAYER_OUT, out_channels=SECOND_LAYER_OUT,
                               kernel_size=SECOND_LAYER_KERNEL_SIZE, stride=SECOND_LAYER_STRIDE)
        self.conv2_bn = nn.BatchNorm2d(SECOND_LAYER_OUT)

        self.conv3 = nn.Conv2d(in_channels=SECOND_LAYER_OUT, out_channels=THIRD_LAYER_OUT,
                               kernel_size=THIRD_LAYER_KERNEL_SIZE, stride=THIRD_LAYER_STRIDE)
        self.conv3_bn = nn.BatchNorm2d(THIRD_LAYER_OUT)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # calculate the output size of the last conv layer
        conv_width = conv2d_size_out(
            conv2d_size_out(
                conv2d_size_out(frame_dim[1], kernel_size=FIRST_LAYER_KERNEL_SIZE, stride=FIRST_LAYER_STRIDE),
                kernel_size=SECOND_LAYER_KERNEL_SIZE, stride=SECOND_LAYER_STRIDE),
            kernel_size=THIRD_LAYER_KERNEL_SIZE, stride=THIRD_LAYER_STRIDE)
        conv_height = conv2d_size_out(
            conv2d_size_out(
                conv2d_size_out(frame_dim[0], kernel_size=FIRST_LAYER_KERNEL_SIZE, stride=FIRST_LAYER_STRIDE),
                kernel_size=SECOND_LAYER_KERNEL_SIZE, stride=SECOND_LAYER_STRIDE),
            kernel_size=THIRD_LAYER_KERNEL_SIZE, stride=THIRD_LAYER_STRIDE)
        num_neurons = conv_width * conv_height * THIRD_LAYER_OUT
        self.fc1 = nn.Linear(in_features=num_neurons, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_actions)

    def forward(self, x):
        out = self.conv1(x)
        conv1 = F.elu(self.conv1_bn(out))

        out = self.conv2(conv1)
        conv2 = F.elu(self.conv2_bn(out))

        out = self.conv3(conv2)
        conv3 = F.elu(self.conv3_bn(out))
        out = conv3.view(conv3.size()[0], -1)

        out = F.elu(self.fc1(out))
        out = F.softmax(self.fc2(out), dim=-1)

        return out, conv1.detach(), conv2.detach(), conv3.detach()


def plot_activation_maps(state, conv_activations, file_name):
    fig = plt.figure(figsize=(64, 64))
    columns = int(math.ceil(math.sqrt(conv_activations.size(0))))
    rows = int(math.ceil(math.sqrt(conv_activations.size(0)))) + 1

    # display the current state in the first row
    fig.add_subplot(rows, columns, 1)
    plt.imshow(state[0].t(), cmap="gray")

    for i in range(0, conv_activations.size(0)):
        # normalize and transpose the image
        img = conv_activations[i] - conv_activations[i].min()
        img = img / img.max()
        img = img.t()

        # add it to the plot
        fig.add_subplot(rows, columns, columns + 1 + i)
        plt.imshow(img)

    plt.savefig(file_name)


env = create_environment()

policy = Policy(env.action_space.n, FRAME_DIM)
policy.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
policy.eval()

state = lazy_frame_to_tensor(env.reset())

for step in count(1):
    predicted_action_values, conv1, conv2, conv3 = policy.forward(state)
    action_dist = Categorical(predicted_action_values)
    action = action_dist.sample().item()

    next_state, reward, done, info = env.step(action)
    next_state = lazy_frame_to_tensor(next_state)

    if step % VISUALIZATION_INTERVAL == 0:
        plot_activation_maps(state.squeeze(), conv1.squeeze(), "activation_visualizations/conv1_{}.png".format(step))
        plot_activation_maps(state.squeeze(), conv2.squeeze(), "activation_visualizations/conv2_{}.png".format(step))
        plot_activation_maps(state.squeeze(), conv3.squeeze(), "activation_visualizations/conv3_{}.png".format(step))

    env.render()

    if done:
        break

    state = next_state
