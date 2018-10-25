import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.saved_rewards = []

    def forward(self, input):
        tmp = self.fc1(input)
        tmp = F.relu(tmp)
        action_scores = self.action_head(tmp)
        action_scores = F.softmax(action_scores, dim=-1)
        state_values = self.value_head(tmp)
        return action_scores, state_values

