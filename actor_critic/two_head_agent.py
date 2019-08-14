import torch
from torch import optim
from torch.distributions import Categorical
from torch.functional import F
import matplotlib.pyplot as plt

from actor_critic.model import ActorCriticNet


class TwoHeadAgent:

    def __init__(self, num_actions, frame_dim, gamma, beta, zeta, lr, device, model_path=""):
        self.device = device
        self.gamma = gamma
        self.zeta = zeta
        self.beta = beta

        self.actor_loss_history = []
        self.critic_loss_history = []

        self.model = ActorCriticNet(num_actions=num_actions, frame_dim=frame_dim).to(device)
        if model_path is not "":
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_action(self, state):
        """Returns a sampled action from the actor network based on the given state."""
        action_values, _ = self.model.forward(state)
        action_distribution = F.softmax(action_values, dim=0)
        probs = Categorical(action_distribution)

        return probs.sample().cpu().detach().item()

    def compute_critic_loss(self, trajectory, predicted_values):
        rewards = [transition[1] for transition in trajectory]

        # discount the rewards
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.FloatTensor(discounted_rewards)

        value_targets = (torch.FloatTensor(rewards).view(-1, 1) + discounted_rewards.view(-1, 1)).to(self.device)

        value_loss = F.mse_loss(predicted_values, value_targets.detach()) * self.zeta

        advantages = value_targets - predicted_values

        return value_loss, advantages.detach()

    def compute_actor_loss(self, trajectory, advantages, predicted_action_values):
        actions = torch.IntTensor([transition[1] for transition in trajectory]).view(-1, 1).to(self.device)

        # get the distribution for each output
        action_distribution = F.softmax(predicted_action_values, dim=1)
        # convert to a categorical distribution to sample from it
        probabilities = Categorical(action_distribution)

        # compute the policy loss
        policy_loss = probabilities.log_prob(actions.view(actions.size(0))).view(-1, 1) * advantages
        policy_loss = -policy_loss.mean()

        entropy = []
        for dist in action_distribution:
            entropy.append(-torch.sum(dist.mean() * torch.log(dist)))
        entropy = torch.stack(entropy).sum() * self.beta

        total_policy_loss = policy_loss - entropy
        return total_policy_loss

    def update(self, trajectory):
        """Updates the actor and critic based on the trajectory."""
        states = torch.cat([transition[0] for transition in trajectory]).to(self.device)
        predicted_action_values, predicted_reward_values = self.model.forward(states)

        critic_loss, advantages = self.compute_critic_loss(trajectory, predicted_reward_values)
        self.critic_loss_history.append(critic_loss)

        actor_loss = self.compute_actor_loss(trajectory, advantages, predicted_action_values)
        self.actor_loss_history.append(actor_loss)

        self.optimizer.zero_grad()
        loss = critic_loss + actor_loss
        loss.backward()
        self.optimizer.step()

        return critic_loss.cpu().detach().item(), actor_loss.cpu().detach().item()

    def plot_loss(self):
        actor_loss_history = torch.FloatTensor(self.actor_loss_history)
        critic_loss_history = torch.FloatTensor(self.critic_loss_history)

        plt.plot(actor_loss_history.numpy(), "y-")
        plt.ylabel("Loss")
        plt.xlabel("Episodes")
        plt.title("Actor Loss")
        plt.show()

        plt.plot(critic_loss_history.numpy(), "m-")
        plt.ylabel("Loss")
        plt.xlabel("Episodes")
        plt.title("Critic Loss")
        plt.show()
