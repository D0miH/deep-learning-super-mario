import torch
from torch import optim
from torch.distributions import Categorical
from torch.functional import F
from torch import nn
import matplotlib.pyplot as plt

import numpy as np

from actor_critic.model import ActorCriticNet


class Agent:

    def __init__(self, num_actions, frame_dim, gamma, beta, zeta, lr, device):
        self.device = device
        self.gamma = gamma
        self.beta = beta
        self.zeta = zeta
        self.lr = lr

        self.loss_history = []

        self.model = ActorCriticNet(num_actions=num_actions, frame_dim=frame_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_action(self, state):
        """Returns a sampled action from the actor network based on the given state."""
        self.model.eval()
        action_values, _ = self.model.forward(state)
        action_distribution = F.softmax(action_values, dim=0)
        probs = Categorical(action_distribution)
        self.model.train()

        return probs.sample().cpu().detach().item()

    def compute_loss(self, trajectory):
        states = torch.cat([transition[0] for transition in trajectory]).to(self.device)
        actions = torch.IntTensor([transition[1] for transition in trajectory]).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor([transition[2] for transition in trajectory]).view(-1, 1).to(self.device)

        # compute the discounted rewards
        discounted_rewards = [
            torch.sum(
                torch.FloatTensor([self.gamma ** i for i in range(rewards[j:].size(0))]).to(self.device) * rewards[j:])
            for j in
            range(rewards.size(0))]

        value_targets = rewards.view(-1, 1) + torch.FloatTensor(discounted_rewards).view(-1, 1).to(self.device)

        # get the predictions of both networks
        action_values, values = self.model.forward(states)
        dists = F.softmax(action_values, dim=1)
        probs = Categorical(dists)

        # compute the value loss (loss of the critic)
        value_loss = F.mse_loss(values, value_targets.detach()) * self.zeta

        # compute the entropy to choose random actions from time to time
        entropy = []
        for dist in dists:
            entropy.append(-torch.sum(dist.mean() * torch.log(dist)))
        entropy = torch.stack(entropy).sum() * self.beta

        # compute the policy loss (loss of the actor)
        advantage = value_targets - values
        policy_loss = -probs.log_prob(actions.view(actions.size(0))).view(-1, 1) * advantage.detach()
        policy_loss = policy_loss.mean()

        total_loss = policy_loss + value_loss - entropy

        self.loss_history.append(total_loss)

        print(
            "Policy Loss: {:.2f}\tValue Loss: {:.2f}\tEntropy: {:.2f}\tTotal Loss: {:.2f}".format(policy_loss.item(),
                                                                                                  value_loss.item(),
                                                                                                  entropy,
                                                                                                  total_loss.item()))

        return total_loss

    def update(self, trajectory):
        """Updates the agent based on the trajectory."""
        loss = self.compute_loss(trajectory)

        self.optimizer.zero_grad()
        loss.backward()

        # clip the gradient to prevent exploding gradients
        # nn.utils.clip_grad_norm_(self.model.parameters(), 5)

        self.optimizer.step()

    def plot_loss(self):
        loss_history = torch.FloatTensor(self.loss_history)
        normalized_loss = (loss_history - loss_history.mean()) / loss_history.std()
        plt.plot(normalized_loss.numpy(), "y-")
        plt.ylabel("Loss")
        plt.xlabel("Episodes")
        plt.legend(["Loss per episode"])
        plt.show()
