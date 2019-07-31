import torch

from CartPoleExample.cart_pole_model import PolicyNet

# hyperparameters
num_episode = 1000  # number of episodes to play
learning_rate = 0.01
gamma = 0.99  # the discount factor for the reward
optimizer = torch.optim.Adam

# other options
render_game = False


def main():
    # create the model
    policy_net = PolicyNet(gamma=gamma, optimizer=optimizer, learning_rate=learning_rate)

    for episode in range(num_episode):
        # render the game in the last 10 episodes
        if episode > num_episode - 10:
            policy_net.play_episode(render=True)
        else:
            policy_net.play_episode(render=render_game)

        # update the weights based on the played episode
        policy_net.train_on_played_episode()

        # plot the graphs every 20 episodes
        if episode % 20 == 0:
            policy_net.plot_durations()


if __name__ == '__main__':
    main()
