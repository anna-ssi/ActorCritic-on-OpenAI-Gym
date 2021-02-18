import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def weights_init_(m):
    """Policy weights initialization."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Critic(nn.Module):
    """Value function approximator."""

    def __init__(self, inp_size, hidden_size):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(inp_size + 1, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class Actor(nn.Module):
    """Deterministic policy gradient."""

    def __init__(self, inp_size, hidden_size, num_actions):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(inp_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        actions = F.softmax(x, dim=-1)
        return actions

    def sample(self, state):
        probs = self.forward(state)

        m = Categorical(probs)
        action = m.sample()
        log_action = m.log_prob(action)

        return action, log_action.unsqueeze(1)

    def to(self, device):
        return super(Actor, self).to(device)
