import torch
from torch.distributions import Categorical

from torch import optim
from torch.functional import F

from actor_critic.model import ActorNet, CriticNet, ActorCriticNet


class TwoHeadAgent:

    def __init__(self, frame_dim, action_space_size, lr, gamma, entropy_scaling, device):
        self.action_space_size = action_space_size
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.entropy_scaling = entropy_scaling

        self.model = ActorCriticNet(action_space_size, frame_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_action(self, state):
        tensor_state = state.to(self.device)

        action_logits, _ = self.model.forward(tensor_state)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)

        return dist.sample().cpu().detach().item()

    def compute_critic_loss(self, trajectory):
        states = [transition[0] for transition in trajectory]
        rewards = [transition[2] for transition in trajectory]
        dones = [transition[3] for transition in trajectory]

        discounted_rewards = []
        cumulative_reward = 0
        for step in reversed(range(len(rewards))):
            cumulative_reward = rewards[step] + self.gamma * cumulative_reward * (1 - int(dones[step]))
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)

        target_values = torch.FloatTensor(rewards).view(-1, 1).to(self.device) + discounted_rewards.view(-1, 1)

        states = torch.cat(states).to(self.device)
        _, actual_values = self.model.forward(states)

        critic_loss = F.l1_loss(actual_values, target_values.view(-1, 1))
        advantage = target_values - actual_values

        return critic_loss, advantage.detach()

    def compute_actor_loss(self, trajectory, advantages):
        states = torch.cat([transition[0] for transition in trajectory]).to(self.device)
        actions = torch.FloatTensor([transition[1] for transition in trajectory]).to(self.device)

        action_logits, _ = self.model.forward(states)
        action_probs = F.softmax(action_logits, dim=1)
        action_dists = Categorical(action_probs)

        # compute the entropy
        entropy = action_dists.entropy().sum()

        policy_loss = -action_dists.log_prob(actions).view(-1, 1) * advantages
        policy_loss = policy_loss.mean()

        return policy_loss - self.entropy_scaling * entropy

    def update(self, trajectory):
        critic_loss, advantage = self.compute_critic_loss(trajectory)

        actor_loss = self.compute_actor_loss(trajectory, advantage)

        total_loss = critic_loss + actor_loss

        torch.cuda.empty_cache()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return actor_loss.cpu().detach(), critic_loss.cpu().detach()

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


class TwoNetAgent:

    def __init__(self, frame_dim, action_space_size, lr_actor, lr_critic, gamma, entropy_scaling, device):
        self.action_space_size = action_space_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.device = device
        self.entropy_scaling = entropy_scaling

        # create the actor net
        self.actor_model = ActorNet(action_space_size, frame_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=lr_actor)

        # create the critic net
        self.critic_model = CriticNet(action_space_size, frame_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=lr_critic)

    def get_action(self, state):
        tensor_state = state.to(self.device)

        action_logits = self.actor_model.forward(tensor_state)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)

        return dist.sample().cpu().detach().item()

    def compute_critic_loss(self, trajectory):
        states = [transition[0] for transition in trajectory]
        rewards = [transition[2] for transition in trajectory]
        dones = [transition[3] for transition in trajectory]

        discounted_rewards = []
        cumulative_reward = 0
        for step in reversed(range(len(rewards))):
            cumulative_reward = rewards[step] + self.gamma * cumulative_reward * (1 - int(dones[step]))
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)

        target_values = torch.FloatTensor(rewards).view(-1, 1).to(self.device) + discounted_rewards.view(-1, 1)

        states = torch.cat(states).to(self.device)
        actual_values = self.critic_model.forward(states)

        critic_loss = F.l1_loss(actual_values, target_values.view(-1, 1))
        advantage = target_values - actual_values

        return critic_loss, advantage.detach()

    def compute_actor_loss(self, trajectory, advantages):
        states = torch.cat([transition[0] for transition in trajectory]).to(self.device)
        actions = torch.FloatTensor([transition[1] for transition in trajectory]).to(self.device)

        action_logits = self.actor_model.forward(states)
        action_probs = F.softmax(action_logits, dim=1)
        action_dists = Categorical(action_probs)

        # compute the entropy
        entropy = action_dists.entropy().sum()

        policy_loss = -action_dists.log_prob(actions).view(-1, 1) * advantages
        policy_loss = policy_loss.mean()

        return policy_loss - self.entropy_scaling * entropy

    def update(self, trajectory):
        torch.cuda.empty_cache()

        critic_loss, advantage = self.compute_critic_loss(trajectory)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        torch.cuda.empty_cache()

        actor_loss = self.compute_actor_loss(trajectory, advantage)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.cpu().detach(), critic_loss.cpu().detach()

    def save_model(self, actor_model_path, critic_model_path):
        torch.save(self.actor_model.state_dict(), actor_model_path)
        torch.save(self.critic_model.state_dict(), critic_model_path)

    def load_model(self, actor_model_path, critic_model_path):
        self.actor_model.load_state_dict(torch.load(actor_model_path))
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.lr_actor)

        self.critic_model.load_state_dict(torch.load(critic_model_path))
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.lr_critic)
