import os
import numpy as np
import pickle
import sys

from rl_agents.agents.common.factory import load_environment

this_dir = os.path.dirname(os.path.abspath(__file__))   
env = load_environment(os.path.join(this_dir, 'games', 'cross_merge_configs', 'env_no_actors_2lane.json'))


def display_episode(b):
    n = len(b.action_history)
    for i in range(n):
        print(f"=======STEP {i}=========")
        print(b.observation_history[i][...,213:].reshape(3,7,10))
        print(b.action_history[i])
        print(b.reward_history[i])

if len(sys.argv) > 1:
    replays = pickle.load(open(sys.argv[1], 'rb'))
    buf = replays['buffer']
    bufs = list(buf.values())

    rews = [x.reward_history for x in bufs]
    rets = [sum(x) for x in rews]


n2 = 0

for _ in range(10000):
    end = False
    rews = 0
    env.reset()
    while not end:
        obs, rew, end, others = env.step(np.random.randint(5))
        rews += rew
    if rews > 1:
        n2 += 1

    print(rews)

print(n2)

import pdb; pdb.set_trace()

for i, a in enumerate(actions):
    #if i == 13:
    #    import pdb; pdb.set_trace()
    obs, rew, end, others = env.step(a)
    print(f"=======STEP {i}=========")
    print(obs[...,213:].reshape(3,7,10))
    print(rew)

