from torch import optim
from torch.distributions import Categorical
import torch
import numpy as np

from policy_gradient.model import Policy


class Agent:

    def __init__(self, num_actions, frame_dim, lr, gamma, device, model_path=""):
        self.lr = lr
        self.gamma = gamma

        self.device = device

        self.model = Policy(num_actions, frame_dim).to(device)
        if model_path is not "":
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action_based_on_state(self, state):
        """Returns the sampled action and the log of the probability density."""
        # get the probability distribution of the prediction
        self.model.eval()
        pred_action_probs = self.model.forward(state)
        pred_action_dist = Categorical(pred_action_probs)

        # sample an action from the probability distribution
        action = pred_action_dist.sample()
        self.model.train()
        return action.item(), pred_action_dist.log_prob(action)

    def update(self, log_prob_history, rewards):
        cumulative_reward = 0
        policy_loss = []
        discounted_rewards = []
        for r in reversed(rewards):
            cumulative_reward = r + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        # normalize the returns
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std() + np.finfo(
            np.float32).eps.item()

        for log_prob, cumulative_reward in zip(log_prob_history, discounted_rewards):
            policy_loss.append(-log_prob * cumulative_reward)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum().to(self.device)
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss.cpu().detach().item()
