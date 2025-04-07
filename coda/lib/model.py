from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING


@dataclass
class L96Parametrized:
    """Hydra config for Lorenz 96 simulator parametrized by Neural Network."""

    _target_: str = "coda.model.lorenz96.L96Parametrized"
    _recursive_: bool = True
    forcing: Any = 8
    parametrization: Any = MISSING
    method: str = "rk4"
    options: Any = None


@dataclass
class FullyConvolutionalNetwork:
    """Hydra config for Fully Convolutional Network."""

    _target_: str = "coda.model.fully_convolutional.FullyConvolutionalNetwork"
    _recursive_: bool = False
    layers: Any = MISSING
    convolution: Any = MISSING
    activation: Any = MISSING
    batch_norm: Any = None
    dropout: Any = None
