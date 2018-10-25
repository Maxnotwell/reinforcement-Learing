import argparse
import gym
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from collections import namedtuple
from Policy_cartpole import Policy
from itertools import count
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--lr', default=0.0003, help='learning rate(default: 0.003')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

envs = gym.make('CartPole-v0')

Saved_Action = namedtuple('SavedAction', ['log_prob', 'value'])

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=args.lr)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    prob, value = policy(state)
    m = Categorical(prob)
    action = m.sample()
    policy.saved_actions.append(Saved_Action(m.log_prob(action), value))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    value_loss = []
    rewards = []
    saved_actions = policy.saved_actions

    for r in policy.saved_rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)

    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_loss.append(-log_prob * reward)
        value_loss.append(F.smooth_l1_loss(value, torch.tensor([r])))

    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
    loss.backward()
    optimizer.step()
    del policy.saved_actions[:]
    del policy.saved_rewards[:]


def main():
    current_reward = 10
    for i_episode in count(1):
        state = envs.reset()
        for j in range(1000):
            action = select_action(state)
            state, reward, done, _ = envs.step(action)
            if args.render:
                envs.render()
            policy.saved_rewards.append(reward)
            if done:
                break

        current_reward = 0.99 * current_reward + 0.01 * j
        finish_episode()

        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, j, current_reward))
        if current_reward > envs.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(current_reward, j))
            break


if __name__ == '__main__':
    main()