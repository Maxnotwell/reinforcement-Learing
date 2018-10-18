import numpy as np
from env import Env
import time

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
MAX_STEP = 30

e = Env()
Q = np.zeros((e.state_num, 4))

def epsilon_greed(Q, state):
    if (np.random.uniform() > 1 - EPSILON) or ((Q[state, :] == 0).all()):
        action = np.random.randint(0, 4)
    else:
        action = Q[state, :].argmax()
    return action

total_reward = 0

for i in range(100):
    e = Env()
    while(e.is_end is False) and (e.step < MAX_STEP):
        action = epsilon_greed(Q, e.present_state)
        state = e.present_state
        reward = e.interact(action)
        total_reward += reward
        new_state = e.present_state
        Q[state, action] = (1 - ALPHA) * Q[state, action] + ALPHA * (reward + GAMMA * Q[new_state, :].max())
        print('Episode:', i, 'Total Step:', e.step, 'Total Reward:', total_reward)
        e.print_map()
        if i > 30:
            time.sleep(0.5)