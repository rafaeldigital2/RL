# -*- coding: utf-8 -*-
"""Cab_RL.py

Book Supplement:
Fundamentals of Reinforcement Learning by Rafael Ris-Ala

# Reinforcement Learning with Q-Learning

# 1.0 Getting to know the environment
"""

!pip install gym==0.21.0

import gym
import random

env = gym.make('Taxi-v3').env

env.reset()

print(env.render(mode='ansi'))

# 0 = south, 1 = north, 2 = east, 3 = west, 4 = pickup, 5 = dropoff
print("Total number of actions: {}".format(env.action_space))

print("Total number of states: {}".format(env.observation_space))

env.P

# env.encode(cab_row, cab_column, passenger_departure, passenger_disembarkation)
env.encode(4, 2, 2, 3)
# This conformation is the state: 451

env.s = 451
print(env.render(mode='ansi'))

# Reward table "P", with states and actions
# This dictionary has the structure {action: [(probability, nextstate, reward, done)]} . That is:
# action 0 (south) has probability 1 of running this action, nextstate 451, reward -1 and if it has reached the end.
# Action 1 (north) has probability 1 to running this action, next state 351, reward -1, and has reached the end.
# Action 2 (east) has probability 1 of running this action, next state 451, reward -1, and has reached the end.
# Action 3 (west) has probability 1 of running this action, next state 431, reward -1, and has reached the end.
# Action 4 (pickup) has probability 1 of running this action, next state 451, reward -10, and has reached the end.
# action 5 (dropoff) has probability 1 to running this action, next state 451, reward -10, and has reached the end.
#env.P[451]

env.P[451]

env.s = 479
print(env.render(mode='ansi'))

"""Note: From state 451 until the taxi reaches its goal, it takes a total of 15 steps. Notice that on the 14th step, the taxi reaches state 479. Note that so far it has got -14 points (14 x -1) and still needs to take the action "drop off the passenger" to reach the final state 475 (True) and get the +20 reward. Thus he gets the total reward of 6, i.e. -14 + 20 = 6.

"""

env.P[479]

"""# 2.0 Taking random actions (without intelligence)

"""

env.s = 484  

epochs = 0   
penalties = 0   

frames = [] 

done = False

while not done:
    action = env.action_space.sample()  
    state, reward, done, info = env.step(action)  

    if reward == -10:  
        penalties += 1
    
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
        }
    )

    epochs += 1

print("Total amount of executed actions: {}".format(epochs))
print("Total amount of penalties received: {}".format(penalties))

"""## 2.1 Showing the animation of the movements executed

"""

from IPython.display import clear_output
from time import sleep

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Step: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)
        
print_frames(frames)

"""# 3.0 Training with the algorithm Q-Learning

Q-learning:

![image.png](data:image/png;base64,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)
"""

import numpy as np
q_table = np.zeros([env.observation_space.n, env.action_space.n])
q_table.shape

q_table

q_table[451]

alpha = 0.1     
gamma = 0.6     
epsilon = 0.1

# Commented out IPython magic to ensure Python compatibility.
# %%time
# for i in range(100000):
#   state = env.reset()
# 
#   penalties, reward = 0, 0
#   done = False
#   while not done:
#     # When the value of "random.uniform(0, 1)" is less than 0.1 (epsilon),    
#     if random.uniform(0, 1) < epsilon:
#       # will execute a random action (Exploration).
#       action = env.action_space.sample()
#     # otherwise, 
#     else:
#       # will execute the action with the highest value, according to q_table
#       # and informed state (Exploitation)
#       action = np.argmax(q_table[state])
# 
#     next_state, reward, done, info = env.step(action)
# 
#     old_q = q_table[state, action]
#     next_max = np.max(q_table[next_state])
# 
#     new_q = (1 - alpha) * old_q + alpha * (reward + gamma * next_max)
#     q_table[state, action] = new_q
# 
#     if reward == -10:
#       penalties += 1
# 
#     state = next_state
# 
#   if i % 100 == 0:
#     clear_output(wait=True)
#     print('Episode: ', i)
# 
# print('Training completed')

q_table

q_table[451]

"""# 4.0 Testing the Q-table
(Now the Q-table is already learned!)
"""

env.s = 451
print(env.render(mode='ansi'))

# 0 = south, 1 = north, 2 = east, 3 = west, 4 = pickup, 5 = dropoff
q_table[451]

env.step(1)
print(env.render(mode='ansi'))

print(env.s)

q_table[env.s]

env.step(1)
print(env.render(mode='ansi'))

print(env.s)

q_table[env.s]

env.step(3)
print(env.render(mode='ansi'))

q_table[env.s]

env.step(3)
print(env.render(mode='ansi'))

q_table[env.s]

env.step(0)
print(env.render(mode='ansi'))

"""Moving on to the 14th step:

And so the taxi follows the biggest rewards... That is until it picks up the passenger and drops him off at his destination.
"""

env.s = 479
print(env.render(mode='ansi'))

q_table[479]

env.step(5)
print(env.render(mode='ansi'))

print(env.s)

q_table[475]

"""# 5.0 Testing the trained agent
(Solving the problem with the learning gained (q_table))
"""

total_penalties = 0
episodes = 50
frames = []

for ep in range(episodes):
  state = env.reset()
  penalties, reward = 0, 0
  done = False
  while not done:
    action = np.argmax(q_table[state])
    state, reward, done, info = env.step(action)

    if reward == -10:
      penalties += 1
    
    frames.append({
        'frame': env.render(mode='ansi'),
        'episode': ep,
        'state': state,
        'action': action,
        'reward': reward
    })

  total_penalties += penalties

print('Episodes', episodes)
print('Penalties', total_penalties)

frames

frames[0]

from time import sleep

for frame in frames:
  clear_output(wait=True)
  print(frame['frame'])
  print(f"Episode: {frame['episode']}")
  print('State:', frame['state'])
  print('Action:', frame['action'])
  print('Reward:', frame['reward'])
  sleep(1)

