import argparse

import gym
from hyperopt import hp, fmin, tpe, space_eval

from agent import Agent
from monitor import interact

parser = argparse.ArgumentParser(description='Hyper-opt settings')
parser.add_argument('--n_iters', type=int, default=5,
                    help='number of independent gym runs to get a mean for set of hyper parameters')
parser.add_argument('--algo', default='sarsamax', help='algo to asses (sarsa, sarsamax, exp_sarsa)')
parser.add_argument('--taxi_version', default='v2', help='algo to asses (sarsa, sarsamax, exp_sarsa)')

c_args = parser.parse_args()


def objective(args):
    env = gym.make(f'Taxi-{c_args.taxi_version}')

    best_scores = []
    for i in range(c_args.n_iters):
        agent = Agent(
            algorithm=c_args.algo,
            alpha=args[0],
            start_epsilon=args[1],
            epsilon_decay=args[2],
            epsilon_cut=None if args[3][0] is None else args[3][1],
            gamma=args[4])

        avg_rewards, best_avg_reward = interact(env, agent, print_logs=False)
        best_scores.append(best_avg_reward)

    return -sum(best_scores) / len(best_scores)


# define a search space

space = [hp.uniform('alpha', 0, 0.3),
         hp.uniform('epsilon_start', 0.5, 1),
         hp.uniform('epsilon_decay', 0.8, 0.9999),
         hp.choice('epsilon_cut', [
             ('epsilon_cut_none', None),
             ('epsilon_cut', hp.uniform('epsilon_cut_value', 0, 0.3))
         ]),
         hp.uniform('gamma', 0.3, 1)]

best = fmin(objective, space, algo=tpe.suggest, max_evals=100)
print('Best params')
print(best)
print('Eval space')
print(space_eval(space, best))
