import distrax
import jax.numpy as jp
import flax.linen as nn

from jaxrl_m.jax_typing import *
from typing import Optional, Sequence, Union, Tuple, Any


def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, mode='fan_avg', distribution='uniform')


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jp.ndarray], jp.ndarray] = nn.relu
    activate_final: int = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    def setup(self):
        self.layers = [nn.Dense(size, kernel_init=self.kernel_init) for size in self.hidden_dims]

    def __call__(self, x: jp.ndarray):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers) or self.activate_final:
                x = self.activations(x)
        return x


class DiscreteCritic(nn.Module):
    hidden_dims: Sequence[int]
    n_actions: int
    activations: Callable[[jp.ndarray], jp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jp.ndarray):
        return MLP((*self.hidden_dims, self.n_actions), activations=self.activations)(observations)


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jp.ndarray], jp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jp.ndarray, actions: jp.ndarray):
        joined = jp.concatenate([observations, actions], axis=-1)
        q_value = MLP((*self.hidden_dims, 1), activations=self.activations)(joined)
        return jp.squeeze(q_value, axis=-1)


def ensemblize(cls, num_critics, out_axes=0, **kwargs):
    return nn.vmap(cls,
                   variable_axes={'params': 0},
                   split_rngs={'params': True},
                   in_axes=None,
                   out_axes=out_axes,
                   axis_size=num_critics,
                   **kwargs)


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jp.ndarray):
        v_value = MLP((*self.hidden_dims, 1))(observations)
        return jp.squeeze(v_value, axis=-1)


class Policy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -20.0
    log_std_max: Optional[float] = 2.0
    tanh_squash: bool = False
    state_dependent_std: bool = True
    final_fc_init_scale: float = 0.01

    @nn.compact
    def __call__(self, observations: jp.ndarray, temperature: float = 1.0) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims, activate_final=True)(observations)
        means = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))(outputs)
        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))(outputs)
        else:
            log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))

        log_stds = jp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, bijector=distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class DiscretePolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    final_fc_init_scale: float = 0.01

    @nn.compact
    def __call__(self, observations: jp.ndarray, temperature: float = 1.0) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims, activate_final=True)(observations)
        logits = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))(outputs)
        distribution = distrax.Categorical(logits=logits / jp.maximum(1e-6, temperature))
        return distribution


class TransformedWithMode(distrax.Transformed):
    def mode(self) -> jp.ndarray:
        return self.bijector.forward(self.distribution.mode())


def get_latent(encoder: nn.Module, observations: Union[jp.ndarray, Dict[str, jp.ndarray]]) -> jp.ndarray:
    if encoder is None:
        return observations

    if isinstance(observations, dict):
        return jp.concatenate([encoder(observations["image"]), observations["state"]], axis=-1)

    return encoder(observations)


class WithEncoder(nn.Module):
    encoder: nn.Module
    network: nn.Module

    def __call__(self, observations: jp.ndarray, *args, **kwargs):
        latents = get_latent(self.encoder, observations)
        return self.network(latents, *args, **kwargs)


class ActorCritic(nn.Module):
    encoders: Dict[str, nn.Module]
    networks: Dict[str, nn.Module]

    def actor(self, observations: jp.ndarray, **kwargs) -> distrax.Distribution:
        latents = get_latent(self.encoders["actor"], observations)
        return self.networks["actor"](latents, **kwargs)

    def critic(self, observations: jp.ndarray, actions: jp.ndarray, **kwargs) -> distrax.Distribution:
        latents = get_latent(self.encoders["critic"], observations)
        return self.networks["critic"](latents, actions, **kwargs)

    def value(self, observations: jp.ndarray, **kwargs) -> distrax.Distribution:
        latents = get_latent(self.encoders["value"], observations)
        return self.networks["value"](latents, **kwargs)

    def __call__(self, observations: jp.ndarray, actions: jp.ndarray):
        rets = {}
        if "actor" in self.networks:
            rets["actor"] = self.actor(observations)
        if "critic" in self.networks:
            rets["critic"] = self.critic(observations, actions)
        if "value" in self.networks:
            rets["value"] = self.value(observations)
        return rets
