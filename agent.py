import numpy as np
import random
import pdb
from collections import defaultdict

class Agent:

    def __init__(self, algorythm='sarsamax', start_epsilon=1, epsilon_decay=0.9, epsilon_cut=0.1, alpha=0.01, gamma=1, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        
        algos = {
            'sarsa': None,
            'sarsamax': self.step_sarmamax,
            'exp_sarsa': None
        }
        
        self.step = algos[algorythm]
        self.Q = defaultdict(lambda: np.zeros(self.nA))    
        self.epsilon, self.epsilon_decay, self.epsilon_cut, self.alpha, self.gamma, self.nA = start_epsilon, epsilon_decay, epsilon_cut, alpha, gamma, nA 
    

    def select_action(self, state):
        r = random.random()
        if r > self.epsilon: # select greedy action with probability epsilon
            return np.argmax(self.Q[state])
        else:                # otherwise, select an action randomly
            return random.randint(0,5)
        

    def step_sarmamax(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if not done: 
            self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])
        else:
            self.Q[state][action] += self.alpha * (reward  - self.Q[state][action])
            self.epsilon = self.epsilon * self.epsilon_decay
            if self.epsilon_cut is not None:
                self.epsilon = max(self.epsilon, self.epsilon_cut)