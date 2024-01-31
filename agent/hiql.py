import copy

import jax
import jax.numpy as jp
import numpy as np

import flax
# flax.config.update('flax_return_frozendict', True)
import flax.linen as nn
import optax
import ml_collections

from jaxrl_m.jax_typing import *
from jaxrl_m.common import TrainState, target_update
from jaxrl_m.networks import Policy, DiscretePolicy, Critic, ensemblize
from agent.iql import IQLAgent
from utils.special_networks import HierarchicalActorCritic, MonolithicVF, Representation, RelativeRepresentation


def expectile_loss(adv, diff: jp.ndarray, tau: float = 0.8):
    weight = jp.where(adv >= 0, tau, 1 - tau)
    return weight * (diff ** 2)


def compute_actor_loss(agent: "HIQLAgent", batch, params):
    use_waypoints = agent.config['use_waypoints']   # Use waypoint states as goals for hierarchical policies

    if use_waypoints:
        subgoals = batch['low_goals']      # subgoal states
    else:
        subgoals = batch['high_goals']     # true goals

    v1, v2 = agent.network(batch['observations'], subgoals, method='value')
    next_v1, next_v2 = agent.network(batch['next_observations'], subgoals, method='value')

    v = (v1 + v2) / 2
    next_v = (next_v1 + next_v2) / 2
    adv = next_v - v
    exp_adv = jp.exp(adv * agent.config['temperature'])
    exp_adv = jp.minimum(exp_adv, 100.0)

    goal_rep_grad = agent.config['policy_train_rep'] if use_waypoints else True
    dist = agent.network(batch['observations'], subgoals,
                         state_rep_grad=True, goal_rep_grad=goal_rep_grad, method='actor', params=params)
    log_probs = dist.log_prob(batch['actions'])
    actor_loss = -(exp_adv * log_probs).mean()

    return actor_loss, {
        'actor_loss': actor_loss,
        'adv': adv.mean(),
        'bc_log_probs': log_probs.mean(),
        'adv_median': jp.median(adv),
        'mse': jp.mean((dist.mode() - batch['actions']) ** 2),
    }


def compute_high_actor_loss(agent: "HIQLAgent", batch, params):
    goals = batch['high_goals']
    subgoals = batch['high_targets']

    v1, v2 = agent.network(batch['observations'], goals, method='value')
    next_v1, next_v2 = agent.network(subgoals, goals, method='value')

    v = (v1 + v2) / 2
    next_v = (next_v1 + next_v2) / 2
    adv = next_v - v
    exp_adv = jp.exp(adv * agent.config['high_temperature'])
    exp_adv = jp.minimum(exp_adv, 100.0)

    dist = agent.network(batch['observations'], goals,
                         state_rep_grad=True, goal_rep_grad=True, method='high_actor', params=params)

    if agent.config['use_rep']:
        target = agent.network(targets=subgoals, bases=batch['observations'], method='value_goal_encoder')
    else:
        target = subgoals - batch['observations']

    log_probs = dist.log_prob(target)
    actor_loss = -(exp_adv * log_probs).mean()

    return actor_loss, {
        'high_actor_loss': actor_loss,
        'high_adv': adv.mean(),
        'high_bc_log_probs': log_probs.mean(),
        'high_adv_median': jp.median(adv),
        'high_mse': jp.mean((dist.mode() - target) ** 2),
        'high_scale': dist.scale_diag.mean()
    }


def compute_value_loss(agent: "HIQLAgent", batch, params):
    batch['masks'] = 1.0 - batch['rewards']
    batch['rewards'] = batch['rewards'] - 1.0

    next_v1_tgt, next_v2_tgt = agent.network(batch['next_observations'], batch['goals'], method='target_value')
    next_v = jp.minimum(next_v1_tgt, next_v2_tgt)
    q_tgt = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v

    v1_tgt, v2_tgt = agent.network(batch['observations'], batch['goals'], method='target_value')
    v_tgt = (v1_tgt + v2_tgt) / 2
    adv = q_tgt - v_tgt

    q1 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v1_tgt
    q2 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v2_tgt
    v1, v2 = agent.network(batch['observations'], batch['goals'], method='value', params=params)

    value_loss1 = expectile_loss(adv, q1 - v1, agent.config['pretrain_expectile']).mean()
    value_loss2 = expectile_loss(adv, q2 - v2, agent.config['pretrain_expectile']).mean()
    value_loss = value_loss1 + value_loss2

    return value_loss, {
        'value_loss': value_loss,
        'v_max': v1.max(),
        'v_min': v1.min(),
        'v_mean': v1.mean(),
        'abs_adv_mean': jp.abs(adv).mean(),
        'adv_mean': adv.mean(),
        'adv_max': adv.max(),
        'adv_min': adv.min(),
        'accept_prob': (adv >= 0).mean()
    }


class HIQLAgent(IQLAgent):
    network: TrainState = None

    def pretrain_update(self,
                        pretrain_batch,
                        seed=None,
                        value_update=True, actor_update=True, high_actor_update=True):

        def loss_fn(params):
            info = {}
            value_loss = actor_loss = high_actor_loss = 0

            if value_update:
                value_loss, value_info = compute_value_loss(self, pretrain_batch, params)
                for k, v in value_info.items():
                    info[f'value/{k}'] = v

            if actor_update:
                actor_loss, actor_info = compute_actor_loss(self, pretrain_batch, params)
                for k, v in actor_info.items():
                    info[f'actor/{k}'] = v

            if high_actor_update and self.config['use_waypoints']:
                high_actor_loss, high_actor_info = compute_high_actor_loss(self, pretrain_batch, params)
                for k, v in high_actor_info.items():
                    info[f'high_actor/{k}'] = v

            loss = value_loss + actor_loss + high_actor_loss
            return loss, info

        alpha = self.config['target_update_rate']
        if value_update:
            new_target_params = jax.tree_map(
                lambda p, target_p: p * alpha + target_p * (1 - alpha),
                self.network.params['networks_value'], self.network.params['networks_target_value']
            )

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn, has_aux=True)
        if value_update:
            # params = flax.core.unfreeze(new_network.params)
            # params['networks_target_value'] = new_target_params
            # new_network = new_network.replace(params=flax.core.freeze(params))
            params = new_network.params.copy()
            params['networks_target_value'] = new_target_params
            new_network = new_network.replace(params=params)

        return self.replace(network=new_network), info

    def sample_actions(self,
                       observations: np.ndarray,
                       goals: np.ndarray,
                       *,
                       low_dim_goals: bool = False,
                       seed: PRNGKey,
                       temperature: float = 1.0,
                       discrete: BoolVec = 0,
                       num_samples: int = None) -> jp.ndarray:

        dist = self.network(observations, goals,
                            low_dim_goals=low_dim_goals, temperature=temperature, method='actor')
        if num_samples is None:
            actions = dist.sample(seed=seed)
        else:
            actions = dist.sample(seed=seed, sample_shape=num_samples)
        if not discrete:
            actions = jp.clip(actions, -1.0, 1.0)

        return actions

    def sample_high_actions(self,
                            observations: np.ndarray,
                            goals: np.ndarray,
                            *,
                            seed: PRNGKey,
                            temperature: float = 1.0,
                            num_samples: int = None) -> jp.ndarray:

        dist = self.network(observations, goals, temperature=temperature, method='high_actor')
        if num_samples is None:
            actions = dist.sample(seed=seed)
        else:
            actions = dist.sample(seed=seed, sample_shape=num_samples)

        return actions

    def get_policy_rep(self, *, targets: np.ndarray, bases: np.ndarray = None):
        return self.network(targets, bases, method='policy_goal_encoder')

    pretrain_update = jax.jit(pretrain_update, static_argnames=('value_update', 'actor_update', 'high_actor_update'))
    sample_actions = jax.jit(sample_actions, static_argnames=('low_dim_goals', 'discrete', 'num_samples'))
    sample_high_actions = jax.jit(sample_high_actions, static_argnames=('num_samples',))
    get_policy_rep = jax.jit(get_policy_rep)

def create_learner(seed: int,
                   observations: jp.ndarray,
                   actions: jp.ndarray,
                   lr: float = 3e-4,
                   actor_hidden_dims: Sequence[int] = (256, 256),
                   value_hidden_dims: Sequence[int] = (256, 256),
                   discount: float = 0.99,
                   alpha: float = 0.005,
                   temperature: float = 1.0,
                   high_temperature: float = 1.0,
                   pretrain_expectile: float = 0.7,
                   way_steps: int = 0,
                   latent_dim: int = 10,
                   use_rep: bool = True,
                   policy_train_rep: BoolVec = 0,
                   visual: BoolVec = 0,
                   encoder: str = 'impala',
                   rep_type: str = 'state',
                   discrete: BoolVec = 0,
                   layer_norm: BoolVec = 0,
                   use_waypoints: BoolVec = 0,
                   **kwargs):

    print('Extra kwargs: ', kwargs)

    rng = jax.random.PRNGKey(seed)
    rng, actor_key, high_actor_key, critic_key, value_key = jax.random.split(rng, 5)

    value_state_encoder = None
    value_goal_encoder = None
    policy_state_encoder = None
    policy_goal_encoder = None
    high_policy_state_encoder = None
    high_policy_goal_encoder = None

    visual_encoder = None
    if visual:
        assert use_rep, 'Visual observations require a representation'
        from jaxrl_m.vision import encoders
        visual_encoder = encoders[encoder]

    def make_encoder(bottleneck: BoolVec):
        if bottleneck:
            hidden_dims = (latent_dim, ) if visual else (*value_hidden_dims, latent_dim)
            return RelativeRepresentation(latent_dim=latent_dim,
                                          hidden_dims=hidden_dims,
                                          visual=visual,
                                          module=visual_encoder,
                                          layer_norm=layer_norm,
                                          rep_type=rep_type,
                                          bottleneck=True)
        else:
            hidden_dims = (value_hidden_dims[-1], ) if visual else (*value_hidden_dims, value_hidden_dims[-1])
            return RelativeRepresentation(latent_dim=value_hidden_dims[-1],
                                          hidden_dims=hidden_dims,
                                          visual=visual,
                                          module=visual_encoder,
                                          layer_norm=layer_norm,
                                          rep_type=rep_type,
                                          bottleneck=False)

    if visual:
        value_state_encoder = make_encoder(bottleneck=False)
        value_goal_encoder = make_encoder(bottleneck=use_waypoints)
        policy_state_encoder = make_encoder(bottleneck=False)
        policy_goal_encoder = make_encoder(bottleneck=False)
        high_policy_state_encoder = make_encoder(bottleneck=False)
        high_policy_goal_encoder = make_encoder(bottleneck=False)

    # ============= Value function =============
    value_def = MonolithicVF(hidden_dims=value_hidden_dims, layer_norm=layer_norm, latent_dim=latent_dim)

    # ============= Actor =============
    if discrete:
        action_dim = actions[0] + 1
        actor_def = DiscretePolicy(hidden_dims=actor_hidden_dims, action_dim=action_dim)
    else:
        action_dim = actions.shape[-1]
        actor_def = Policy(hidden_dims=actor_hidden_dims, action_dim=action_dim,
                           log_std_min=-5.0, state_dependent_std=False, tanh_squash=False)

    high_action_dim = latent_dim if use_rep else observations.shape[-1]
    high_actor_def = Policy(hidden_dims=actor_hidden_dims, action_dim=high_action_dim,
                            log_std_min=-5.0, state_dependent_std=False, tanh_squash=False)

    # ============= Network =============
    network_def = HierarchicalActorCritic(
        encoders={'value_state': value_state_encoder,
                  'value_goal': value_goal_encoder,
                  'policy_state': policy_state_encoder,
                  'policy_goal': policy_goal_encoder,
                  'high_policy_state': high_policy_state_encoder,
                  'high_policy_goal': high_policy_goal_encoder},
        networks={'value': value_def,
                  'target_value': copy.deepcopy(value_def),
                  'actor': actor_def,
                  'high_actor': high_actor_def},
        use_waypoints=use_waypoints,
    )

    network_tx = optax.chain(optax.zero_nans(), optax.adam(learning_rate=lr))
    network_params = network_def.init(value_key, observations, goals=observations)['params']
    network = TrainState.create(network_def, network_params, tx=network_tx)
    # params = flax.core.unfreeze(network.params)
    # params['networks_target_value'] = params['networks_value']  # Copy value network to target network
    # network = network.replace(params=flax.core.freeze(params))  # Replace target network params
    params = network.params.copy()
    params['networks_target_value'] = params['networks_value']  # Copy value network to target network
    network = network.replace(params=params)  # Replace target network params

    config = flax.core.FrozenDict({
        'discount': discount, 'temperature': temperature, 'high_temperature': high_temperature,
        'target_update_rate': alpha, 'pretrain_expectile': pretrain_expectile, 'way_steps': way_steps,
        'latent_dim': latent_dim, 'policy_train_rep': policy_train_rep, 'use_rep': use_rep,
        'use_waypoints': use_waypoints      # whether to use subgoal states as goals for hierarchical policies
    })

    return HIQLAgent(rng, network=network, critic=None, value=None, target_value=None, actor=None, config=config)


def get_default_config():
    config = ml_collections.ConfigDict({
        'lr': 3e-4,
        'actor_hidden_dims': (256, 256),
        'value_hidden_dims': (256, 256),
        'discount': 0.99,
        'alpha': 0.005,
        'pretrain_expectile': 0.7,
        'temperature': 1.0,
    })

    return config
