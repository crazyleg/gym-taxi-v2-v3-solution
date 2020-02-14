from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v2')
agent = Agent(algorithm='exp_sarsa')
avg_rewards, best_avg_reward = interact(env, agent)