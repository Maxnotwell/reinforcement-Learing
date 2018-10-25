import torch
import torch.nn as nn
import argparse
import gym
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from itertools import count
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='policy grandient')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor(default 0.99)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--lr', default=0.01, help='learning rate(default 0.01')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make('CartPole-v0')


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.prob_log = []
        self.reward_log = []

    def forward(self, input):
        out = self.fc1(input)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.softmax(out, dim=1)

        return out


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=args.lr)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    prob = policy(state)
    m = Categorical(prob)
    action = m.sample()
    policy.prob_log.append(m.log_prob(action))
    return action.item()


def finish_episode():
    A = 0
    policy_loss = []
    advantage = []
    for r in policy.reward_log[::-1]:
        A = r + args.gamma * A
        advantage.insert(0, A)

    advantage = torch.tensor(advantage)
    advantage = (advantage - advantage.mean()) / (advantage.std() + eps)

    for prob, reward in zip(policy.prob_log, advantage):
        policy_loss.append(-prob * reward)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.reward_log[:]
    del policy.prob_log[:]


def main():
    current_reward = 10
    for i in count(1):
        state = env.reset()
        for j in range(10000):
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.reward_log.append(reward)
            if done:
                break

        current_reward = current_reward * 0.99 + j * 0.01
        finish_episode()
        if i % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i, j, current_reward))
        if current_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(current_reward, j))
            break


if __name__ == '__main__':
    main()
