from omegaconf import OmegaConf
import torch
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    NewType,
    Optional,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
)
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

# Tensor dtype
# for jaxtyping usage, see https://github.com/google/jaxtyping/blob/main/API.md
from jaxtyping import Bool, Complex, Float, Inexact, Int, Integer, Num, Shaped, UInt

# Config type
from omegaconf import DictConfig

# PyTorch Tensor type
from torch import Tensor

# Runtime type checking decorator
from typeguard import typechecked as typechecker


def broadcast(tensor, src=0):
    if not _distributed_available():
        return tensor
    else:
        torch.distributed.broadcast(tensor, src=src)
        return tensor

def _distributed_available():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    # added by Xavier -- delete '--local-rank' in multi-nodes training, don't know why there is such a keyword
    if '--local-rank' in cfg:
        del cfg['--local-rank']
    # added by Xavier -- delete '--local-rank' in multi-nodes training, don't know why there is such a keyword
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg