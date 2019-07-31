

import torch

from mario_model import MarioNet

num_episodes = 100
learning_rate = 0.01
gamma = 0.99
optimizer = torch.optim.Adam

mario_model = MarioNet(gamma=gamma, optimizer=optimizer, learning_rate=learning_rate)

for episode in range(num_episodes):
    print("In episode:", episode)

    mario_model.play_episode(render=False)

    mario_model.train_on_played_episode()


# play two games after training
mario_model.play_episode(render=True)
mario_model.play_episode(render=True)

