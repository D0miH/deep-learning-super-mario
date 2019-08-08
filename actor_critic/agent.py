import torch
from torch import optim
from torch.distributions import Categorical
from torch.functional import F
import matplotlib.pyplot as plt

from actor_critic.model import ActorNet, CriticNet


class Agent:

    def __init__(self, num_actions, frame_dim, gamma, beta, zeta, actor_lr, critic_lr, device):
        self.device = device
        self.gamma = gamma
        self.beta = beta
        self.zeta = zeta

        self.actor_loss_history = []
        self.critic_loss_history = []

        self.actor_model = ActorNet(num_actions=num_actions, frame_dim=frame_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=actor_lr)

        self.critic_model = CriticNet(frame_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=critic_lr)

    def get_action(self, state):
        """Returns a sampled action from the actor network based on the given state."""
        self.actor_model.eval()
        action_values = self.actor_model.forward(state)
        action_distribution = F.softmax(action_values, dim=0)
        probs = Categorical(action_distribution)
        self.actor_model.train()

        return probs.sample().cpu().detach().item()

    def compute_critic_loss(self, trajectory):
        states = torch.cat([transition[0] for transition in trajectory]).to(self.device)
        rewards = [transition[1] for transition in trajectory]

        # discount the rewards
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.FloatTensor(discounted_rewards)

        value_targets = (torch.FloatTensor(rewards).view(-1, 1) + discounted_rewards.view(-1, 1)).to(self.device)

        # get the predicted values
        predicted_values = self.critic_model.forward(states)

        value_loss = F.mse_loss(predicted_values, value_targets.detach()) * self.zeta

        advantages = value_targets - predicted_values

        return value_loss, advantages.detach()

    def compute_actor_loss(self, trajectory, advantages):
        states = torch.cat([transition[0] for transition in trajectory]).to(self.device)
        actions = torch.IntTensor([transition[1] for transition in trajectory]).view(-1, 1).to(self.device)

        # get the predicted actions
        predicted_action_values = self.actor_model.forward(states)
        # get the distribution for each output
        action_distribution = F.softmax(predicted_action_values, dim=1)
        # convert to a categorical distribution to sample from it
        probabilities = Categorical(action_distribution)

        # calculate the entropy of the action distributions
        entropy = []
        for dist in action_distribution:
            entropy.append(-torch.sum(dist.mean() * torch.log(dist)))
        entropy = torch.stack(entropy).sum() * self.beta

        # compute the policy loss
        policy_loss = probabilities.log_prob(actions.view(actions.size(0))).view(-1, 1) * advantages
        policy_loss = -policy_loss.mean()

        total_policy_loss = policy_loss - entropy
        return total_policy_loss

    def update(self, trajectory):
        """Updates the actor and critic based on the trajectory."""
        critic_loss, advantages = self.compute_critic_loss(trajectory)
        self.critic_loss_history.append(critic_loss)

        actor_loss = self.compute_actor_loss(trajectory, advantages)
        self.actor_loss_history.append(actor_loss)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return critic_loss, actor_loss

    def plot_loss(self):
        actor_loss_history = torch.FloatTensor(self.actor_loss_history)
        critic_loss_history = torch.FloatTensor(self.critic_loss_history)

        plt.plot(actor_loss_history.numpy(), "y-", critic_loss_history.numpy(), "m-")
        plt.ylabel("Loss")
        plt.xlabel("Episodes")
        plt.legend(["Actor Loss per Episode", "Critic Loss per Episode"])
        plt.show()
