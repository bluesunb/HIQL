from jaxrl_m.jax_typing import *

import jax
import jax.numpy as jp
import numpy as np
import optax
from jaxrl_m.common import TrainState, target_update
from jaxrl_m.networks import Policy, Critic, ValueCritic, ensemblize

import flax
from flax import struct
import ml_collections


def expectile_loss(diff: jp.ndarray, tau: float = 0.8):
    weight = jp.where(diff > 0, tau, 1 - tau)
    return weight * (diff ** 2)


class IQLAgent(struct.PyTreeNode):
    rng: PRNGKey
    critic: TrainState
    value: TrainState
    target_value: TrainState
    actor: TrainState
    config: dict = struct.field(pytree_node=False)

    @jax.jit
    def update(self, batch: Batch, seed=None) -> InfoDict:
        def critic_loss_fn(params):
            next_v = self.target_value(batch['next_observations'])
            target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
            q1, q2 = self.critic(batch['observations'], batch['actions'], params=params)
            critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
            return critic_loss, {
                'critic_loss': critic_loss,
                'q1': q1.mean(),
            }

        def value_loss_fn(params):
            q1, q2 = self.critic(batch['observations'], batch['actions'])
            q = jp.minimum(q1, q2)
            v = self.value(batch['observations'], params=params)
            value_loss = expectile_loss(q - v, tau=self.config['tau']).mean()
            advantage = q - v
            return value_loss, {
                'value_loss': value_loss,
                'v': v.mean(),
                'abs_adv_mean': jp.abs(advantage).mean(),
                'adv_mean': advantage.mean(),
                'adv_max': advantage.max(),
                'adv_min': advantage.min(),
            }

        def actor_loss_fn(params):
            v = self.value(batch['observations'])
            q1, q2 = self.critic(batch['observations'], batch['actions'])
            q = jp.minimum(q1, q2)
            exp_adv = jp.exp((q - v) * self.config['temperature'])
            exp_adv = jp.minimum(exp_adv, 100.0)

            dist = self.actor(batch['observations'], params=params)
            log_probs = dist.log_prob(batch['actions'])
            actor_loss = -(exp_adv * log_probs).mean()

            sorted_adv = jp.sort(q - v)[::-1]
            return actor_loss, {
                'actor_loss': actor_loss,
                'adv': q - v,
                'bc_log_probs': log_probs.mean(),
                'adv_median': jp.median(q - v),
                'adv_top_1%': sorted_adv[int(0.01 * len(sorted_adv))],
                'adv_top_10%': sorted_adv[int(0.1 * len(sorted_adv))],
                'adv_top_25%': sorted_adv[int(0.25 * len(sorted_adv))],
                'adv_top_50%': sorted_adv[int(0.5 * len(sorted_adv))],
                'adv_top_75%': sorted_adv[int(0.75 * len(sorted_adv))],
            }

        new_critic, critic_info = self.critic.apply_loss_fn(loss_fn=critic_loss_fn, has_aux=True)
        new_target_value = target_update(self.value, self.target_value, self.config['target_update_rate'])
        new_value, value_info = self.value.apply_loss_fn(loss_fn=value_loss_fn, has_aux=True)
        new_actor, actor_info = self.actor.apply_loss_fn(loss_fn=actor_loss_fn, has_aux=True)

        return (self.replace(critic=new_critic, target_value=new_target_value, value=new_value, actor=new_actor),
                {**critic_info, **value_info, **actor_info})
    

@jax.jit
def sample_actions(agent: IQLAgent,
                   observations: jp.ndarray,
                   *,
                   seed: PRNGKey,
                   temperature: float = 1.0) -> jp.ndarray:
    
    actions = agent.actor(observations, temperature=temperature).sample(seed=seed)
    actions = jp.clip(actions, -1.0, 1.0)
    return actions


def create_learner(seed: int,
                   observations: jp.ndarray,
                   actions: jp.ndarray,
                   hidden_dims: Sequence[int] = (256, 256),
                   actor_lr: float = 3e-4, value_lr: float = 3e-4, critic_lr: float = 3e-4,
                   discount: float = 0.99,
                   alpha: float = 0.005,    # target update rate
                   tau: float = 0.8,        # expectile
                   temperature: float = 0.1,
                   dropout_rate: Optional[float] = None,
                   max_steps: Optional[int] = None,
                   opt_decay_schedule: str = 'cosine',
                   **kwargs):
    
    print('Extra kwargs: ', kwargs)

    rng = jax.random.PRNGKey(seed)
    rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

    def make_actor():
        actor_def = Policy(hidden_dims,
                           action_dim=actions.shape[-1],
                           log_std_min=-5.0,
                           state_dependent_std=False,
                           tanh_squash=False)
        
        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(init_value=-actor_lr, decay_steps=max_steps)
            actor_tx = optax.chain(optax.scale_by_adam(),
                                   optax.scale_by_schedule(schedule_fn))
        else:
            actor_tx = optax.adam(learning_rate=actor_lr)
        
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(actor_def, actor_params, tx=actor_tx)
        return actor
    
    def make_critic():
        critic_def = ensemblize(Critic, num_critics=2)(hidden_dims)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic_tx = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(critic_def, critic_params, tx=critic_tx)
        return critic
    
    def make_value():
        value_def = ValueCritic(hidden_dims=hidden_dims)
        value_params = value_def.init(value_key, observations)["params"]
        value_tx = optax.adam(learning_rate=value_lr)
        value = TrainState.create(value_def, value_params, tx=value_tx)
        target_value = TrainState.create(value_def, value_params)   # no optimizer for target
        return value, target_value

    actor = make_actor()
    critic = make_critic()
    value, target_value = make_value()

    config = flax.core.FrozenDict({
        "discount": discount,
        "temperature": temperature,
        "tau": tau,
        "target_update_rate": alpha,
    })

    return IQLAgent(rng, critic, value, target_value, actor, config)


def get_default_configs():
    config = ml_collections.ConfigDict({
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'value_lr': 3e-4,
        'hidden_dims': (256, 256),
        'discount': 0.99,
        'tau': 0.9,
        'alpha': 0.005,
        'temperature': 10.0,
        'dropout_rate': None,
    })

    return config


if __name__ == "__main__":
    learner = create_learner(seed=0,
                             observations=jp.zeros((32, 17)),
                             actions=jp.zeros((32, 4)),
                             max_steps=1000)

    print(learner)