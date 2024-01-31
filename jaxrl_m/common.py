import jax
import jax.numpy as jp
import flax.linen as nn
from flax import struct
import optax
from functools import partial

from jaxrl_m.jax_typing import *
from typing import Callable, Optional

nonpytree_fields = partial(struct.field, pytree_node=False)


def shard_batch(batch: Batch):
    """
    Split a batch into sub-batches for each device.
    (e.g. (128, ...) -> (n_devices, 128 // n_devices, ...))
    """
    n_devices = jax.local_device_count()
    def reshape(x):
        assert x.shape[0] % n_devices == 0, f"Batch size {x.shape[0]} not divisible by {n_devices}"
        return x.reshape((n_devices, x.shape[0] // n_devices, *x.shape[1:]))
    return jax.tree_map(reshape, batch)


def target_update(model: "TrainState", target_model: "TrainState", tau: float) -> "TrainState":
    """Soft-update of target model parameters with Polyak averaging."""
    new_target_params = jax.tree_map(
        lambda p, target_p: p * tau + target_p * (1 - tau),
        model.params, target_model.params
    )
    return target_model.replace(params=new_target_params)


# from flax.training.train_state import TrainState as FlaxTrainState
class TrainState(struct.PyTreeNode):
    step: int   # Current training step
    apply_fn: Callable[..., Any] = nonpytree_fields()  # Function to apply to get model output
    model_def: Any = nonpytree_fields()  # Model definition
    params: Params
    tx: Optional[optax.GradientTransformation] = nonpytree_fields()  # Gradient transformation
    opt_state: Optional[optax.OptState] = None  # Optimizer state

    @classmethod
    def create(cls, 
               model_def: nn.Module, 
               params: Params, 
               tx: Optional[optax.GradientTransformation] = None, 
               **kwargs) -> "TrainState":
        
        opt_state = None
        if tx is not None:
            opt_state = tx.init(params)

        return cls(step=1,
                   apply_fn=model_def.apply,
                   model_def=model_def,
                   params=params,
                   tx=tx,
                   opt_state=opt_state,
                   **kwargs)
    
    def __call__(self, *args, params=None, extra_variables: dict = None, method: ModuleMethod = None, **kwargs):
        if params is None:
            params = self.params
        
        variables = {'params': params}
        if extra_variables is not None:
            variables.update(extra_variables)

        if isinstance(method, str):
            method = getattr(self.model_def, method)

        return self.apply_fn(variables, *args, method=method, **kwargs)
    
    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        
        return self.replace(
            step=self.step + 1, 
            params=new_params, 
            opt_state=new_opt_state, 
            **kwargs
        )
    
    def apply_loss_fn(self, *, loss_fn, pmap_axis=None, has_aux=False):
        if has_aux:
            grads, info = jax.grad(loss_fn, has_aux=True)(self.params)
            if pmap_axis is not None:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis)
                info = jax.lax.pmean(info, axis_name=pmap_axis)
            return self.apply_gradients(grads=grads), info
        
        else:
            grads = jax.grad(loss_fn)(self.params)
            if pmap_axis is not None:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis)
            return self.apply_gradients(grads=grads)