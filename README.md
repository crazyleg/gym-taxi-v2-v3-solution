# Solution for OpenAI Gym Taxi problem v2 and v3 using temporal difference methods - SarsaMax and Expected Sarsa
This is a solution for Gym Taxi problem as discussed in the Reinforcement Learning cource at Udacity.

main.py and monitor.py are slightly modified versions for enviroment setup from the cource.

agent.py is my solution for the problem
hyper_opt.py is a script to find optimal set of hyperparameters for each algorithm.


## Attention - Taxt-V2 vs Taxi-V3

Recent version of Gym has deprecated Taxi-v2, which was mainly used to it's leaderboard. So, by default local requirements.txt install gym==0.14, that still has Taxi-V2.

Taxi-V3 is a more difficult version of the problem, to run it, please do manually

```bash
pip install gym==0.16
```

To run hyper parameters optimization you can use 

```bash
python hyper_opt.py --taxi_version v3
```

## SarsaMax optimal performance
## ExpectedSarsa optimal performance