from dataclasses import dataclass
from typing import Union


@dataclass
class Four4DVarLoss:
    _target_: str = "coda.utils.Loss4DVar.Four4DVarLoss"
    alpha: Union[float] = 0

@dataclass
class NegativeLogLikelihoodLoss:
    _target_: str = "coda.utils.LossNLL.NegativeLogLikelihoodLoss"
    alpha: Union[float] = 0
    
@dataclass
class NegativeLogLikelihoodLoss_Full:
    _target_: str = "coda.utils.LossNLL_full.NegativeLogLikelihoodLoss_Full"
    alpha: Union[float] = 0