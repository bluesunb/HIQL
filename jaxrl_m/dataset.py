import numpy as np
import jax
from flax.core.frozen_dict import FrozenDict

from jaxrl_m.jax_typing import Data, Array


def get_size(data: Data) -> int:
    sizes = jax.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_leaves(sizes))


class Dataset(FrozenDict):
    """
    A class for storing (and retrieving batches of) data in nested dictionary format.

    Example:
    ```
        dataset = Dataset({
            'observations': {
                'image': np.random.randn(100, 28, 28, 1),
                'state': np.random.randn(100, 4),
            },
            'actions': np.random.randn(100, 2),
        })

        batch = dataset.sample(32)
        # Batch should have nested shape: {
        #   'observations': {'image': (32, 28, 28, 1), 'state': (32, 4)},
        #   'actions': (32, 2)
        # }
    ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)

    @classmethod
    def create(cls,
               observations: Data,
               actions: Array,
               rewards: Array,
               masks: Array,
               next_observations: Data,
               freeze: bool = True,
               **extra_fields) -> "Dataset":
        
        data = {"observations": observations,
                "actions": actions,
                "rewards": rewards,
                "masks": masks,
                "next_observations": next_observations,
                **extra_fields}
        
        if freeze:  # set read-only
            jax.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def sample(self, batch_size: int, index: Array=None):
        if index is None:
            index = np.random.randint(self.size, size=batch_size)
        return self.get_subset(index)
    
    def get_subset(self, index: Array):
        return jax.tree_map(lambda arr: arr[index], self._dict)
    

class ReplayBuffer(Dataset):
    def __init__(self, *args, **kwargs):
        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    @classmethod
    def create(cls, transition: Data, size: int):
        """
        create dict of buffers with the same structure as transition
        """
        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)
        
        buffer_dict = jax.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def creeate_from_initial_dataset(cls, init_dataset: dict, size: int):
        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[:len(init_buffer)] = init_buffer
            return buffer
        
        buffer_dict = jax.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset
    

    def add_transition(self, transition):
        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element
        
        jax.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)
