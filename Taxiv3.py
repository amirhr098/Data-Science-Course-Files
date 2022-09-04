import gym
import random
import numpy as np

env = gym.make("Taxi-v3", render_mode="human")
env.action_space.seed(42)

observation = env.reset(seed=42)
q_table = np.zeros([env.observation_space.n, env.action_space.n])

alpha = 0.1
gamma = 0.6
epsilon = 0.1
all_epochs = []
all_penalties = []
for i in range(1, 10000):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        next_state, reward, done, _ = env.step(action)
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value
        
        if reward == -10:
            penalties += 1
        state = next_state
        epochs += 1
    
    if i % 10 == 0:
        print('Episode: ', i)
        print('Penalties: ', penalties)
        print('Epochs: ', epochs)

env.close()