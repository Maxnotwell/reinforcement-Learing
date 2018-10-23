import time
import numpy as np
from env import Env

total_reward = 0
GAMMA = 0.9
ALPHA = 0.1
EPSILON = 0.2
MAX_STEP = 30

def epsilon_greed(Q, state):
    if (np.random.uniform() > EPSILON) or ((Q[state, :] == 0).all()):
        action = np.random.randint(0, 4)
    else:
        action = Q[state, :].argmax()
    return action

e = Env()
Q = np.zeros((e.state_num, 4))

for i in range(100):
    e = Env()
    while(e.is_end is False) and (e.step < MAX_STEP):
        state = e.present_state
        action = epsilon_greed(Q, state)
        reward = e.interact(action)
        total_reward += reward
        new_state = e.present_state
        new_action = epsilon_greed(Q, new_state)
        Q[state, action] = (1 - ALPHA) * Q[state, action] + ALPHA * (reward + GAMMA * Q[new_state, new_action])
        action = new_action
        print('Episode:', i, 'Total Step:', e.step, 'Total Reward:', total_reward)
        e.print_map()
        time.sleep(0.5)
