import random

import numpy as np
import torch


class ReplayMemory:
    def __init__(self, capacity, seed=31):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, device):
        self.device = device
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, end = map(np.stack, zip(*batch))
        return (self.correct(state), self.correct(action),
                self.correct(reward, unsqueeze=True), self.correct(next_state),
                self.correct(end, unsqueeze=True))

    def correct(self, inp, unsqueeze=False):
        correct = torch.FloatTensor(inp).to(self.device)
        if unsqueeze:
            correct = correct.unsqueeze(1)
        return correct

    def __len__(self):
        return len(self.buffer)
