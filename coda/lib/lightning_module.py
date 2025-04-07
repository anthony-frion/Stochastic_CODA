from dataclasses import dataclass
from typing import Any


@dataclass
class DataAssimilationModule:
    _target_: str = "coda.model.lightning_model.DataAssimilationModule"
    _recursive_: bool = False
    simulator: Any = None
    assimilation_network: Any = None
    rollout_length: int = 5
    time_step: float = 0.01
    loss: Any = None
    optimizer: Any = None
    
    
@dataclass
class DataAssimilationGaussianModule:
    _target_: str = "coda.model.lightning_model_gaussian.DataAssimilationModule"
    _recursive_: bool = False
    simulator: Any = None
    assimilation_network: Any = None
    rollout_length: int = 5
    time_step: float = 0.01
    loss: Any = None
    optimizer: Any = None
    sample_mean: bool = False

@dataclass
class DataAssimilationGaussianLRModule:
    _target_: str = "coda.model.lightning_model_LR_gaussian.DataAssimilationModule"
    _recursive_: bool = False
    simulator: Any = None
    assimilation_network: Any = None
    rollout_length: int = 5
    time_step: float = 0.01
    loss: Any = None
    optimizer: Any = None
    sample_mean: bool = False

@dataclass
class ParameterTuningModule:
    _target_: str = "coda.model.lightning_model.ParameterTuningModule"
    _recursive_: bool = False
    simulator: Any = None
    assimilation_network: Any = None
    rollout_length: int = 5
    time_step: float = 0.01
    loss: Any = None
    optimizer: Any = None


@dataclass
class ParametrizationLearningModule:
    _target_: str = "coda.model.lightning_model.ParametrizationLearningModule"
    _recursive_: bool = False
    simulator: Any = None
    assimilation_network: Any = None
    rollout_length: int = 5
    time_step: float = 0.01
    loss: Any = None
    optimizer: Any = None
