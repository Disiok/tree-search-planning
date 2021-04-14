import numpy as np
import ray
import os
import seaborn as sns
from  matplotlib import pyplot as plt
from sys import argv

from muzero import MuZero

ckpts_dir = 'results/cross_merge_env/ckpts'
envs = ['one', 'one_aggro']
ckpts = ['1lane.ckpt', '1lane_aggro.ckpt']


if argv[1] == 'eval':
    all_results = {}

    for env in envs:
        print(f"====== Evaluating environment: {env} ======\n")
        
        muzero = MuZero('cross_merge_env', env_cfg_key=env, num_simulations=25)
        all_results[env] = {}

        for ckpt in ckpts:
            print(f">>> Evaluating checkpoint: {ckpt}")

            ckpt_pth = os.path.join(ckpts_dir, ckpt)
            muzero.load_model(checkpoint_path=ckpt_pth)

            result = muzero.test(render=False, opponent='self', num_gpus=1, num_tests=400)
            
            n_crashed = [r[-1]['crashed'] for r in result['episode_infos']]
            n_goals = [r[-1]['num_goals_reached'] for r in result['episode_infos']]
            speeds = [sum([r_['speed'] for r_ in r[1:]]) / len(r[1:]) for r in result['episode_infos'] if len(r) > 1]
            rews = [sum([r_['reward'] for r_ in r]) for r in result['episode_infos']]
            lens = [len(r_)for r_ in result['episode_infos']]

            result['n_crashed'] = n_crashed    
            result['speeds'] = speeds
            result['n_goals'] = n_goals
            result['rewards'] = rews
            result['episode_lens'] = lens
            
            result['speeds_std'] = np.std(speeds)
            result['n_goals_std'] = np.std(n_goals)
            result['rewards_std'] = np.std(rews)
            result['episode_lens_std'] = np.std(lens)
            
            mean_speed = np.mean(speeds)
            pct_crash = np.mean(n_crashed)
            mean_goals = np.mean(n_goals)

            result['mean_speed'] = mean_speed
            result['pct_crash'] = pct_crash
            result['mean_goals'] = mean_goals

            result.pop('episode_infos')
            all_results[env][ckpt[:-5]] = result
        
            print(np.mean(rews))

        ray.shutdown()

    import pdb; pdb.set_trace()
    np.save('actor_behavior.npy', all_results)

elif argv[1] == 'plot':
    info = np.load('density_sweep.npy', allow_pickle=True)
    densities = list(info.keys())
    metrics = ['mean_total_reward']
    names = ['sparse', 'dense']
    ckpts = ['1lane_sparse', '1lane_dense']

    for metric in metrics:
        for i, ckpt in enumerae(ckpts):
            metric_vals = [info[d][ckpt][metric] for d in densities]
            sns.plot(densities, metric_vals, label=names[i])





