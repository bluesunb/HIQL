import numpy as np
import jax.numpy as jp
import flax

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union


PRNGKey = jp.ndarray
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
Dtype = Any
InfoDict = Dict[str, float]
Array = Union[np.ndarray, jp.ndarray]
Data = Union[Array, Dict[str, "Data"]]
Batch = Dict[str, Data]
ModuleMethod = Union[str, Callable, None]
BoolVec = Union[bool, int]
