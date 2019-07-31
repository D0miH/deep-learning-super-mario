import torch

from CartPoleExample.cart_pole_model import PolicyNet

# hyperparameters
num_episode = 1000  # number of episodes to play
batch_size = 5  # number of episodes per batch
learning_rate = 0.01
gamma = 0.99  # the discount factor for the reward
optimizer = torch.optim.Adam

# other options
render_game = False


def main():
    # create the model
    policy_net = PolicyNet(gamma=gamma, optimizer=optimizer, learning_rate=learning_rate)

    # train the model
    for episode in range(int(num_episode / batch_size)):
        policy_net.play_episode_batch(batch_size=batch_size, render=render_game)

        # update the weights based on the played episode
        policy_net.train_on_played_episode_batch()

        policy_net.plot_durations()

    # let the model play for two episodes
    policy_net.play_episode(render=True)
    policy_net.play_episode(render=True)

if __name__ == '__main__':
    main()
