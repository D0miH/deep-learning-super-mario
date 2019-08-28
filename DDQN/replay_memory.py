import random


class ReplayMemory(object):
    """Replay memory for learning."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Adds a transition to the replay memory."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample transitions at random."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)