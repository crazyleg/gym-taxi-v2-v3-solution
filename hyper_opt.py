from agent import Agent
from monitor import interact
import gym
import pdb
import numpy as np
import hyperopt
from hyperopt import hp


N_ITERS = 5

def objective(args):
    env = gym.make('Taxi-v2')
    
    best_scores = []
    for i in range(N_ITERS):
        agent = Agent(
            alpha=args[0], 
            start_epsilon=args[1], 
                      epsilon_decay=args[2], 
                      epsilon_cut=None if args[3][0]==None else args[3][1], 
                      gamma=args[4])

        avg_rewards, best_avg_reward = interact(env, agent, print_logs=False)
        best_scores.append(best_avg_reward)

    return -sum(best_scores)/len(best_scores)
    

# define a search space

space = [hp.uniform('alpha', 0, 0.3),
        hp.uniform('epsilon_start', 0.5, 1),
        hp.uniform('epsilon_decay',0.8, 0.9999),
        hp.choice('epsilon_cut', [
            ('epsilon_cut_none', None),
            ('epsilon_cut', hp.uniform('epsilon_cut_value', 0, 0.3))
        ]),
        hp.uniform('gamma', 0.3, 1)]

# minimize the objective over the space
from hyperopt import fmin, tpe, space_eval

best = fmin(objective, space, algo=tpe.suggest, max_evals=100)
print('Best params')
print(best)
print('Eval space')
print(space_eval(space, best))
