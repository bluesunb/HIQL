import os, time
import gzip
from datetime import datetime
from tqdm import tqdm
from functools import partial

import numpy as np
import jax
import jax.numpy as jp
import flax

import pickle
import wandb
from ml_collections import config_flags

from agent import hiql
from utils.gc_dataset import GCSDataset
from utils import d4rl_utils, d4rl_ant, ant_diagnostics, viz_utils
from utils.additional import record_video, CsvLogger
from utils.config import Config

from jaxrl_m.wandb import setup_wandb, default_wandb_config
# from jaxrl_m.evaluation import supply_rng, evaluate_with_trajectories, EpisodeMonitor
from jaxrl_m.evaluation import evaluate_with_trajectories, EpisodeMonitor

# from absl import app, flags
#
# FLAGS = flags.FLAGS
# flags.DEFINE_string('env_name', 'antmaze-large-diverse-v2', '')
# flags.DEFINE_string('save_dir', f'experiment_output/', '')
# flags.DEFINE_string('run_group', 'Debug', '')
# flags.DEFINE_integer('seed', 0, '')
# flags.DEFINE_integer('eval_episodes', 50, '')
# flags.DEFINE_integer('num_video_episodes', 2, '')
# flags.DEFINE_integer('log_interval', 1000, '')
# flags.DEFINE_integer('eval_interval', 100000, '')
# flags.DEFINE_integer('save_interval', 100000, '')
# flags.DEFINE_integer('batch_size', 1024, '')
# flags.DEFINE_integer('pretrain_steps', 0, '')
#
# # flags.DEFINE_integer('layer_norm', 1, '')
# flags.DEFINE_integer('layer_norm', 1, '')
# flags.DEFINE_integer('value_hidden_dim', 512, '')
# flags.DEFINE_integer('value_num_layers', 3, '')
# flags.DEFINE_integer('use_rep', 0, '')
# flags.DEFINE_integer('rep_dim', None, '')
# flags.DEFINE_enum('rep_type', 'state', ['state', 'diff', 'concat'], '')
# flags.DEFINE_integer('policy_train_rep', 0, '')
# flags.DEFINE_integer('use_waypoints', 0, '')
# flags.DEFINE_integer('way_steps', 1, '')
#
# flags.DEFINE_float('pretrain_expectile', 0.7, '')
# flags.DEFINE_float('p_randomgoal', 0.3, '')
# flags.DEFINE_float('p_trajgoal', 0.5, '')
# flags.DEFINE_float('p_currgoal', 0.2, '')
# flags.DEFINE_float('high_p_randomgoal', 0., '')
# flags.DEFINE_integer('geom_sample', 1, '')
# flags.DEFINE_float('discount', 0.99, '')
# flags.DEFINE_float('temperature', 1, '')
# flags.DEFINE_float('high_temperature', 1, '')
#
# flags.DEFINE_integer('visual', 0, '')
# flags.DEFINE_string('encoder', 'impala', '')
#
# flags.DEFINE_string('algo_name', None, '')  # Not used, only for logging
#
# wandb_config = default_wandb_config()
# wandb_config.update({
#     'project': 'hiql',
#     'group': 'Debug',
#     'name': '{env_name}',
# })
#
# config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
# config_flags.DEFINE_config_dict('config', hiql.get_default_config(), lock_config=False)
#
# gcdataset_config = GCSDataset.get_default_config()
# config_flags.DEFINE_config_dict('gcdataset', gcdataset_config, lock_config=False)


@jax.jit
def get_debug_statistics(agent, batch):
    def get_info(s, g):
        return agent.network(s, g, info=True, method='value')

    s = batch['observations']
    g = batch['goals']

    info = get_info(s, g)

    stats = {}

    stats.update({
        'v': info['v'].mean(),
    })

    return stats


@jax.jit
def get_gcvalue(agent, s, g):
    v1, v2 = agent.network(s, g, method='value')
    return (v1 + v2) / 2


def get_v(agent, goal, observations):
    goal = jp.tile(goal, (observations.shape[0], 1))
    return get_gcvalue(agent, observations, goal)


@jax.jit
def get_traj_v(agent, trajectory):
    def get_v(s, g):
        v1, v2 = agent.network(jax.tree_map(lambda x: x[None], s), jax.tree_map(lambda x: x[None], g), method='value')
        return (v1 + v2) / 2

    observations = trajectory['observations']
    all_values = jax.vmap(jax.vmap(get_v, in_axes=(None, 0)), in_axes=(0, None))(observations, observations)
    return {
        'dist_to_beginning': all_values[:, 0],
        'dist_to_end': all_values[:, -1],
        'dist_to_middle': all_values[:, all_values.shape[1] // 2],
    }


def supply_rng(f):
    def wrapped_f(*args, **kwargs):
        rng, key = jax.random.split(jax.random.PRNGKey(0))
        return f(*args, seed=key, **kwargs)
    return wrapped_f


def main(cfg: Config):
    g_start_time = int(datetime.now().timestamp())

    exp_name = ''
    exp_name += f'sd{cfg.seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    if 'SLURM_RESTART_COUNT' in os.environ:
        exp_name += f'rs_{os.environ["SLURM_RESTART_COUNT"]}.'
    exp_name += f'{g_start_time}'
    # exp_name += f'_{cfg.wandb["name"]}'

    cfg.gcdataset['p_randomgoal'] = cfg.p_randomgoal
    cfg.gcdataset['p_trajgoal'] = cfg.p_trajgoal
    cfg.gcdataset['p_currgoal'] = cfg.p_currgoal
    cfg.gcdataset['geom_sample'] = cfg.geom_sample
    cfg.gcdataset['high_p_randomgoal'] = cfg.high_p_randomgoal
    cfg.gcdataset['way_steps'] = cfg.way_steps
    cfg.gcdataset['discount'] = cfg.discount
    cfg.config['pretrain_expectile'] = cfg.pretrain_expectile
    cfg.config['discount'] = cfg.discount
    cfg.config['temperature'] = cfg.temperature
    cfg.config['high_temperature'] = cfg.high_temperature
    cfg.config['use_waypoints'] = cfg.use_waypoints
    cfg.config['way_steps'] = cfg.way_steps
    cfg.config['value_hidden_dims'] = (cfg.value_hidden_dim,) * cfg.value_num_layers
    cfg.config['use_rep'] = cfg.use_rep
    cfg.config['rep_dim'] = cfg.rep_dim
    cfg.config['policy_train_rep'] = cfg.policy_train_rep

    # Create wandb logger
    params_dict = {**cfg.gcdataset.to_dict(), **cfg.config.to_dict()}
    # cfg.wandb['name'] = cfg.wandb['exp_descriptor'] = exp_name
    # cfg.wandb['group'] = cfg.wandb['exp_prefix'] = cfg.run_group
    # setup_wandb(params_dict, **cfg.wandb)

    # cfg.save_dir = os.path.join(cfg.save_dir, wandb.run.project, wandb.config.exp_prefix,
    #                               wandb.config.experiment_id)
    os.makedirs(cfg.save_dir, exist_ok=True)

    goal_info = None
    discrete = False
    if 'antmaze' in cfg.env_name:
        env_name = cfg.env_name

        if 'ultra' in cfg.env_name:
            import d4rl_ext
            import gym
            env = gym.make(env_name)
            env = EpisodeMonitor(env)
        else:
            env = d4rl_utils.make_env(env_name)

        dataset = d4rl_utils.get_dataset(env, cfg.env_name)
        dataset = dataset.copy({'rewards': dataset['rewards'] - 1.0})

        env.render(mode='rgb_array', width=200, height=200)
        if 'large' in cfg.env_name:
            env.viewer.cam.lookat[0] = 18
            env.viewer.cam.lookat[1] = 12
            env.viewer.cam.distance = 50
            env.viewer.cam.elevation = -90

            viz_env, viz_dataset = d4rl_ant.get_env_and_dataset(env_name)
            viz = ant_diagnostics.Visualizer(env_name, viz_env, viz_dataset, discount=cfg.discount)
            init_state = np.copy(viz_dataset['observations'][0])
            init_state[:2] = (12.5, 8)
        elif 'ultra' in cfg.env_name:
            env.viewer.cam.lookat[0] = 26
            env.viewer.cam.lookat[1] = 18
            env.viewer.cam.distance = 70
            env.viewer.cam.elevation = -90
        else:
            env.viewer.cam.lookat[0] = 18
            env.viewer.cam.lookat[1] = 12
            env.viewer.cam.distance = 50
            env.viewer.cam.elevation = -90
    # elif 'kitchen' in cfg.env_name:
    #     env = d4rl_utils.make_env(cfg.env_name)
    #     dataset = d4rl_utils.get_dataset(env, cfg.env_name, filter_terminals=True)
    #     dataset = dataset.copy({'observations': dataset['observations'][:, :30],
    #                             'next_observations': dataset['next_observations'][:, :30]})
    # elif 'calvin' in cfg.env_name:
    #     from src.envs.calvin import CalvinEnv
    #     from hydra import compose, initialize
    #     from src.envs.gym_env import GymWrapper
    #     from src.envs.gym_env import wrap_env
    #     initialize(config_path='src/envs/conf')
    #     cfg = compose(config_name='calvin')
    #     env = CalvinEnv(**cfg)
    #     env.max_episode_steps = cfg.max_episode_steps = 360
    #     env = GymWrapper(
    #         env=env,
    #         from_pixels=cfg.pixel_ob,
    #         from_state=cfg.state_ob,
    #         height=cfg.screen_size[0],
    #         width=cfg.screen_size[1],
    #         channels_first=False,
    #         frame_skip=cfg.action_repeat,
    #         return_state=False,
    #     )
    #     env = wrap_env(env, cfg)
    #
    #     data = pickle.load(gzip.open('data/calvin.gz', "rb"))
    #     ds = []
    #     for i, d in enumerate(data):
    #         if len(d['obs']) < len(d['dones']):
    #             continue  # Skip incomplete trajectories.
    #         # Only use the first 21 states of non-floating objects.
    #         d['obs'] = d['obs'][:, :21]
    #         new_d = dict(
    #             observations=d['obs'][:-1],
    #             next_observations=d['obs'][1:],
    #             actions=d['actions'][:-1],
    #         )
    #         num_steps = new_d['observations'].shape[0]
    #         new_d['rewards'] = np.zeros(num_steps)
    #         new_d['terminals'] = np.zeros(num_steps, dtype=bool)
    #         new_d['terminals'][-1] = True
    #         ds.append(new_d)
    #     dataset = dict()
    #     for key in ds[0].keys():
    #         dataset[key] = np.concatenate([d[key] for d in ds], axis=0)
    #     dataset = d4rl_utils.get_dataset(None, cfg.env_name, dataset=dataset)
    # elif 'procgen' in cfg.env_name:
    #     from src.envs.procgen_env import ProcgenWrappedEnv, get_procgen_dataset
    #     import matplotlib
    #
    #     matplotlib.use('Agg')
    #
    #     n_processes = 1
    #     env_name = 'maze'
    #     env = ProcgenWrappedEnv(n_processes, env_name, 1, 1)
    #
    #     if cfg.env_name == 'procgen-500':
    #         dataset = get_procgen_dataset('data/procgen/level500.npz', state_based=('state' in cfg.env_name))
    #         min_level, max_level = 0, 499
    #     elif cfg.env_name == 'procgen-1000':
    #         dataset = get_procgen_dataset('data/procgen/level1000.npz', state_based=('state' in cfg.env_name))
    #         min_level, max_level = 0, 999
    #     else:
    #         raise NotImplementedError
    #
    #     # Test on large levels having >=20 border states
    #     large_levels = [12, 34, 35, 55, 96, 109, 129, 140, 143, 163, 176, 204, 234, 338, 344, 369, 370, 374, 410, 430,
    #                     468, 470, 476, 491] + [5034, 5046, 5052, 5080, 5082, 5142, 5244, 5245, 5268, 5272, 5283, 5335,
    #                                            5342, 5366, 5375, 5413, 5430, 5474, 5491]
    #     goal_infos = [{'eval_level': [level for level in large_levels if min_level <= level <= max_level],
    #                    'eval_level_name': 'train'},
    #                   {'eval_level': [level for level in large_levels if level > max_level], 'eval_level_name': 'test'}]
    #
    #     dones_float = 1.0 - dataset['masks']
    #     dones_float[-1] = 1.0
    #     dataset = dataset.copy({
    #         'dones_float': dones_float
    #     })
    #
    #     discrete = True
    #     example_action = np.max(dataset['actions'], keepdims=True)
    else:
        raise NotImplementedError

    env.reset()

    pretrain_dataset = GCSDataset(dataset, **cfg.gcdataset.to_dict())
    total_steps = cfg.pretrain_steps
    example_batch = dataset.sample(1)
    agent = hiql.create_learner(cfg.seed,
                                example_batch['observations'],
                                # example_batch['actions'] if not discrete else example_action,
                                example_batch['actions'],
                                visual=cfg.visual,
                                encoder=cfg.encoder,
                                discrete=discrete,
                                layer_norm=cfg.layer_norm,
                                rep_type=cfg.rep_type,
                                **cfg.config)

    # For debugging metrics
    if 'antmaze' in cfg.env_name:
        example_trajectory = pretrain_dataset.sample(50, indx=np.arange(1000, 1050))
    # elif 'kitchen' in cfg.env_name:
    #     example_trajectory = pretrain_dataset.sample(50, indx=np.arange(0, 50))
    # elif 'calvin' in cfg.env_name:
    #     example_trajectory = pretrain_dataset.sample(50, indx=np.arange(0, 50))
    # elif 'procgen-500' in cfg.env_name:
    #     example_trajectory = pretrain_dataset.sample(50, indx=np.arange(5000, 5050))
    # elif 'procgen-1000' in cfg.env_name:
    #     example_trajectory = pretrain_dataset.sample(50, indx=np.arange(5000, 5050))
    else:
        raise NotImplementedError

    train_logger = CsvLogger(os.path.join(cfg.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(cfg.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm(range(1, total_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):
        pretrain_batch = pretrain_dataset.sample(cfg.batch_size)
        jax.debug_nans(True)
        agent, update_info = supply_rng(agent.pretrain_update)(pretrain_batch)
        # agent, update = agent.pretrain_update(pretrain_batch)

        if i % cfg.log_interval == 0:
            debug_statistics = get_debug_statistics(agent, pretrain_batch)
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics.update({f'pretraining/debug/{k}': v for k, v in debug_statistics.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / cfg.log_interval
            train_metrics['time/total_time'] = (time.time() - first_time)
            last_time = time.time()
            # wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        if i == 1 or i % cfg.eval_interval == 0:
            policy_fn = partial(supply_rng(agent.sample_actions), discrete=discrete)
            high_policy_fn = partial(supply_rng(agent.sample_high_actions))
            policy_rep_fn = agent.get_policy_rep
            base_observation = jax.tree_map(lambda arr: arr[0], pretrain_dataset.dataset['observations'])
            # if 'procgen' in cfg.env_name:
            #     eval_metrics = {}
            #     for goal_info in goal_infos:
            #         eval_info, trajs, renders = evaluate_with_trajectories(
            #             policy_fn, high_policy_fn, policy_rep_fn, env, env_name=cfg.env_name,
            #             num_episodes=cfg.eval_episodes,
            #             base_observation=base_observation, num_video_episodes=0,
            #             use_waypoints=cfg.use_waypoints,
            #             eval_temperature=0, epsilon=0.05,
            #             goal_info=goal_info, config=cfg.config,
            #         )
            #         eval_metrics.update(
            #             {f'evaluation/level{goal_info["eval_level_name"]}_{k}': v for k, v in eval_info.items()})
            # else:
            eval_info, trajs, renders = evaluate_with_trajectories(
                policy_fn, high_policy_fn, policy_rep_fn, env, env_name=cfg.env_name,
                num_episodes=cfg.eval_episodes,
                base_observation=base_observation, num_video_episodes=cfg.num_video_episodes,
                use_waypoints=cfg.use_waypoints,
                eval_temperature=0,
                goal_info=goal_info, config=cfg.config,
            )
            eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}

    #         if cfg.num_video_episodes > 0:
    #             video = record_video('Video', i, renders=renders)
    #             eval_metrics['video'] = video
    #
    #         traj_metrics = get_traj_v(agent, example_trajectory)
    #         value_viz = viz_utils.make_visual_no_image(
    #             traj_metrics,
    #             [partial(viz_utils.visualize_metric, metric_name=k) for k in traj_metrics.keys()]
    #         )
    #         # eval_metrics['value_traj_viz'] = wandb.Image(value_viz)
    #
    #         if 'antmaze' in cfg.env_name and 'large' in cfg.env_name and cfg.env_name.startswith('antmaze'):
    #             traj_image = d4rl_ant.trajectory_image(viz_env, viz_dataset, trajs)
    #             # eval_metrics['trajectories'] = wandb.Image(traj_image)
    #
    #             new_metrics_dist = viz.get_distance_metrics(trajs)
    #             eval_metrics.update({
    #                 f'debugging/{k}': v for k, v in new_metrics_dist.items()})
    #
    #             image_v = d4rl_ant.gcvalue_image(
    #                 viz_env,
    #                 viz_dataset,
    #                 partial(get_v, agent),
    #             )
    #             # eval_metrics['v'] = wandb.Image(image_v)
    #
    #         # wandb.log(eval_metrics, step=i)
    #         eval_logger.log(eval_metrics, step=i)
    #
    #     if i % cfg.save_interval == 0:
    #         save_dict = dict(
    #             agent=flax.serialization.to_state_dict(agent),
    #             config=cfg.config.to_dict()
    #         )
    #
    #         fname = os.path.join(cfg.save_dir, f'params_{i}.pkl')
    #         print(f'Saving to {fname}')
    #         with open(fname, "wb") as f:
    #             pickle.dump(save_dict, f)
    # train_logger.close()
    # eval_logger.close()


if __name__ == '__main__':

    cfg = Config(env_name='antmaze-medium-play-v2',
                 run_group='EXP',
                 seed=0,
                 pretrain_steps=500002,
                 eval_episodes=5,
                 eval_interval=50000,
                 save_interval=125000,
                 p_currgoal=0.2,
                 p_trajgoal=0.5,
                 p_randomgoal=0.3,
                 high_p_randomgoal=0.3,
                 discount=0.99,
                 temperature=1.0,
                 high_temperature=1,
                 pretrain_expectile=0.7,
                 geom_sample=1,
                 layer_norm=0,
                 value_hidden_dim=512,
                 value_num_layers=3,
                 batch_size=1024,
                 use_rep=0,
                 policy_train_rep=0,
                 algo_name='hiql',
                 use_waypoints=1,
                 way_steps=25,)
    
    cfg.config = hiql.get_default_config()
    cfg.gcdataset = GCSDataset.get_default_config()

    # argv = config.to_argv()
    # sys.argv.extend(argv)
    # app.run(main)
    main(cfg)