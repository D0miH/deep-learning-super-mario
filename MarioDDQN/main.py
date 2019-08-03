import random
from itertools import count
import matplotlib.pyplot as plt

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

import numpy as np

from MarioDDQN.dq_network import DQNetwork
from MarioDDQN.replay_memory import ReplayMemory
from MarioDDQN.wrappers import wrapper

FRAME_DIM = (84, 84, 4)
ACTION_SPACE = SIMPLE_MOVEMENT
REPLAY_MEMORY_CAPACITY = 100_000
NUM_EPISODES = 10_000
MAX_STEPS_PER_GAME = 1000
GAMMA = 0.99
RENDER_ENVIRONMENT = False

BATCH_SIZE = 32
TRAIN_FREQUENCE = 4  # number of total steps after which the policy model is trained
TARGET_NETWORK_UPDATE_FREQUENCE = 40_000  # number of total steps after which the weights of the policy model are copied to the target model

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_STEPS = 1_000_000
EXPLORATION_DECAY = (EXPLORATION_MAX - EXPLORATION_MIN) / EXPLORATION_STEPS


###################################
#####  CODE IS AT THE BOTTOM  #####
###################################

def get_next_action(state, action_space, current_exploration):
    """Returns the next action."""
    if np.random.rand() < current_exploration:
        return random.randrange(action_space)
    else:
        predicted_q_values = policy_net.model.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0),
                                                      batch_size=1)
        return np.argmax(predicted_q_values[0])


def train_policy_model(replay_memory, policy_model, target_model):
    """Trains the policy net on a batch from the replay memory."""

    # if there are not enough transitions yet don't train
    if len(replay_memory) < BATCH_SIZE:
        return

    batch = replay_memory.sample(BATCH_SIZE)

    # get the target q values of the target model
    current_states = []
    target_predictions = []
    for state, action, reward, next_state, done in batch:
        # convert the states
        current_state = np.expand_dims(np.asarray(state).astype(np.float64), axis=0)
        next_state = np.expand_dims(np.asarray(next_state).astype(np.float64), axis=0)

        # get the prediction of the target network and the current policy network
        target_prediction = target_model.predict(next_state)
        target_q_value = np.max(target_prediction)
        current_prediction = policy_model.predict(current_state)[0]

        # calculate the new q values
        if done:
            current_prediction[action] = reward
        else:
            current_prediction[action] = reward + GAMMA * target_q_value

        current_states.append(current_state)
        target_predictions.append(target_prediction)

    # train the policy model based on the predictions of the target model
    policy_model.fit(np.asarray(current_states).squeeze(), np.asarray(target_predictions).squeeze(),
                     batch_size=BATCH_SIZE, verbose=0)


env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, ACTION_SPACE)
# apply the wrapper
env = wrapper(env)

# create the network
policy_net = DQNetwork(stacked_frame_dim=FRAME_DIM, num_actions=env.action_space.n)
target_net = DQNetwork(stacked_frame_dim=FRAME_DIM, num_actions=env.action_space.n)

# create the replay memory
replay_memory = ReplayMemory(REPLAY_MEMORY_CAPACITY)

# play the episodes
current_exploration = EXPLORATION_MAX
total_steps = 0
reward_history = []
mean_reward_history = []
for episode in range(NUM_EPISODES):
    state = env.reset()

    # play one game
    current_reward = 0
    for steps in count(MAX_STEPS_PER_GAME):
        # render the environment
        if RENDER_ENVIRONMENT:
            env.render()

        # get the next action
        action = get_next_action(state, env.action_space.n, current_exploration)

        # perform the action
        next_state, reward, done, info = env.step(action)

        # if we are done end the loop
        if done or info["life"] < 2:
            print("Episode: ", episode, "reward: ", current_reward)
            reward_history.append(current_reward)
            mean_reward_history.append(np.mean(reward_history))
            break

        # add the transition to the replay memory
        replay_memory.push(state, action, reward, next_state, done)

        # increase the current reward and the total steps
        current_reward += reward
        total_steps += 1

        # train the policy network
        if total_steps % TRAIN_FREQUENCE == 0:
            train_policy_model(replay_memory, policy_net.model, target_net.model)

        # update the weights of the target model if necessary
        if total_steps % TARGET_NETWORK_UPDATE_FREQUENCE == 0:
            print("Updated the target Model")
            target_net.model.set_weights(policy_net.model.get_weights())

        # update the exploration rate
        current_exploration -= EXPLORATION_DECAY
        current_exploration = max(EXPLORATION_MIN, current_exploration)

    # plot the reward history
    if (episode + 1) % 10 == 0 and episode > 100:
        plt.plot(reward_history, "b-", mean_reward_history, "r-")
        plt.ylabel("Rewards")
        plt.xlabel("Episodes")
        plt.show()

