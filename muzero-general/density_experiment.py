import numpy as np
import ray
import os
from sys import argv

from muzero import MuZero

ckpts_dir = 'results/cross_merge_env/ckpts'
env = 'one'
densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ckpts = ['1lane_dense.ckpt', '1lane_sparse.ckpt']


if argv[1] == 'run_eval':
    all_results = {}

    for density in densities:
        print(f"====== Evaluating density: {density} ======\n")
        
        muzero = MuZero('cross_merge_env', env_cfg_key=env, density=density, num_simulations=25)
        all_results[density] = {}

        for ckpt in ckpts:
            print(f">>> Evaluating checkpoint: {ckpt}")

            ckpt_pth = os.path.join(ckpts_dir, ckpt)
            muzero.load_model(checkpoint_path=ckpt_pth)

            result = muzero.test(render=False, opponent='self', num_gpus=1, num_tests=50)
            
            n_crashed = [r[-1]['crashed'] for r in result['episode_infos']]
            n_goals = [r[-1]['num_goals_reached'] for r in result['episode_infos']]
            speeds = [sum([r_['speed'] for r_ in r[1:]]) / len(r[1:]) for r in result['episode_infos']]
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
            
            mean_speed = sum(speeds) / len(speeds)
            pct_crash = sum(n_crashed) / len(result)
            mean_goals = sum(n_goals) /len(result)

            result['mean_speed'] = mean_speed
            result['pct_crash'] = pct_crash
            result['mean_goals'] = mean_goals

            result.pop('episode_infos')
            all_results[density][ckpt[:-5]] = result
        
            print(np.mean(rews))

        ray.shutdown()

    import pdb; pdb.set_trace()
    np.save('density_sweep.npy', all_results)

elif argv[1] == 'plot':

