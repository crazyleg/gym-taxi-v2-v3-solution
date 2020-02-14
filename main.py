from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v2')
agent = Agent(algorithm='exp_sarsa',
              alpha=0.2512238484351891,
              epsilon_cut=0,
              epsilon_decay=0.8888782926665223,
              start_epsilon=0.9957089031634627,
              gamma=0.7749915552696941)

avg_rewards, best_avg_reward = interact(env, agent)