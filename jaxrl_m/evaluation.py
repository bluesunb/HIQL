import time
import gym
import jax
import jax.numpy as jp
import numpy as np
from collections import defaultdict
from typing import Dict, Callable, Any

from agent.hiql import HIQLAgent


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    def wrapper(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)
    return wrapper


def flatten(d: Dict[str, Any], parent_key: str = "", sep: str = "."):
    """
    {a: {b: 1, c: 2}, d: 3} -> {"a.b": 1, "a.c": 2, "d": 3
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """
    {a: [1, 2], b: [3, 4]}, {a: 5, b: 6} -> {a: [1, 2, 5], b: [3, 4, 6]
    """
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def kitchen_render(kitchen_env, wh=64):
    from dm_control.mujoco import engine
    camera = engine.MovableCamera(kitchen_env.sim, wh, wh)
    camera.set_pose(distance=1.8, lookat=[-0.3, 0.5, 2.0], azimuth=90, elevation=60)
    img = camera.render()
    return img


def evaluate(policy_fn, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    stats = defaultdict(list)
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action = policy_fn(observation)
            observation, reward, done, info = env.step(action)
            add_to(stats, flatten(info))
        add_to(stats, flatten(info, parent_key="final"))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats


def evaluate_with_trajectories(policy_fn: HIQLAgent.sample_actions,
                               high_policy_fn: HIQLAgent.sample_high_actions,
                               policy_rep_fn: HIQLAgent.get_policy_rep,
                               env: gym.Env, env_name: str, num_episodes: int,
                               base_observation: np.ndarray = None,
                               num_video_episodes: int = 0,
                               use_waypoints: bool = False,
                               eval_temperature: int = 0,
                               epsilon: float = 0.0,
                               goal_info: Dict[str, Any] = None,
                               config: Any = None) -> Dict[str, float]:
    
    trajectories = []
    stats = defaultdict(list)
    renders = []

    use_rep = config['use_rep']
    for i in range(num_episodes + num_video_episodes):
        trajectory = defaultdict(list)
        observation, done = env.reset(), False

        if 'antmaze' in env_name:
            goal = env.wrapped_env.target_goal
            obs_goal = base_observation.copy()
            obs_goal[:2] = goal
        else:
            raise NotImplementedError(f'{env_name} have to be implemented')
        
        render = []
        step = 0
        while not done:
            if use_waypoints:
                now_obs_goal = high_policy_fn(observation, goals=obs_goal, temperature=eval_temperature)
                if use_rep:
                    now_obs_goal = now_obs_goal / np.linalg.norm(now_obs_goal, axis=-1, keepdims=True) * np.sqrt(now_obs_goal.shape[-1])
                else:
                    now_obs_goal = observation + now_obs_goal
                now_obs_goal_rep = now_obs_goal

            else:
                now_obs_goal = obs_goal
                if use_rep:
                    now_obs_goal_rep = policy_rep_fn(targets=now_obs_goal, bases=observation)
                else:
                    now_obs_goal_rep = now_obs_goal

            action = policy_fn(observations=observation, goals=now_obs_goal_rep, low_dim_goals=True, temperature=eval_temperature)
            if 'antmaze' in env_name:
                next_observation, reward, done, info = env.step(action)
            else:
                raise NotImplementedError(f'{env_name} have to be implemented')
            
            step += 1

            if i >= num_episodes and step % 3 == 0:
                if 'antmaze' in env_name:
                    size = 200
                    now_frame = env.render(mode='rgb_array', width=size, height=size).transpose(2, 0, 1).copy()
                    if use_waypoints and not use_rep and ('large' in env_name or 'ultra' in env_name):
                        def xy_to_pixel_xy(x, y):
                            if 'large' in env_name:
                                pixx = (x / 35) * (0.93 - 0.07) + 0.07
                                pixy = (y / 24) * (0.21 - 0.79) + 0.79
                            elif 'ultra' in env_name:
                                pixx = (x / 52) * (0.955 - 0.05) + 0.05
                                pixy = (y / 36) * (0.19 - 0.81) + 0.81
                            return pixx, pixy
                        
                        def get_slice(v, pad):
                            return slice(int((v - pad) * size), int((v + pad) * size))

                        x, y = now_obs_goal_rep[:2]
                        pixx, pixy = xy_to_pixel_xy(x, y)
                        now_frame[0, get_slice(pixy, 0.02), get_slice(pixx, 0.02)] = 255
                        now_frame[1:3, get_slice(pixy, 0.02), get_slice(pixx, 0.02)] = 0
                    
                    render.append(now_frame)
            
            transition = {'observation': observation,
                          'next_observations': next_observation,
                          'action': action,
                          'reward': reward,
                          'done': done,
                          'info': info}
            
            add_to(trajectory, transition)
            add_to(stats, flatten(info))
            observation = next_observation
        
        add_to(stats, flatten(info, parent_key="final"))
        trajectories.append(trajectory)
        if i >= num_episodes:
            renders.append(render)
        
    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajectories, renders


class EpisodeMonitor(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if done:
            info['episode'] = {'return': self.reward_sum,
                               'length': self.episode_length,
                               'duration': time.time() - self.start_time}
            
            if hasattr(self, 'get_normalized_score'):
                info['episode']['normalized_return'] = self.get_normalized_score(info['episode']['return']) * 100.0

        return observation, reward, done, info
    
    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()
            