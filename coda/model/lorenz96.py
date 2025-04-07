from typing import Union

import torch
import torch.nn as nn
from mdml_tools.simulators.base import BaseSimulator
from mdml_tools.simulators.lorenz96 import l96_tendencies_x


class L96Parametrized(BaseSimulator):
    def __init__(
        self,
        forcing: Union[float, torch.Tensor, nn.Parameter],
        parametrization: nn.Module,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.forcing = forcing
        self.add_module("parametrization", parametrization)

    def forward(self, t: torch.Tensor, state: torch.Tensor):
        tendencies_x = l96_tendencies_x(x=state, forcing=self.forcing)
        return tendencies_x + self.parametrization.forward(state)
