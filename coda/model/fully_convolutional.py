from typing import Optional

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig


class FullyConvolutionalNetwork(nn.Module):
    def __init__(
        self,
        layers: list[int],
        convolution: DictConfig,
        activation: DictConfig,
        batch_norm: Optional[DictConfig] = None,
        dropout: Optional[DictConfig] = None,
    ):
        super().__init__()
        self.layers = layers
        self.conv_factory = instantiate(convolution, _partial_=True)
        self.activation_factory = instantiate(activation, _partial_=True)
        if batch_norm:
            self.batch_norm_factory = instantiate(batch_norm, _partial_=True)
        if dropout:
            self.dropout_factory = instantiate(dropout, _partial_=True)
        self.__setup_model()

    def __setup_model(self):
        use_batch_norm = True if hasattr(self, "batch_norm_factory") else False
        use_dropout = True if hasattr(self, "dropout_factory") else False

        model = []
        model.append(self.conv_factory(in_channels=self.layers[0], out_channels=self.layers[1]))
        for i in range(1, len(self.layers) - 1):
            if use_batch_norm:
                model.append(self.batch_norm_factory(num_features=self.layers[i]))
            model.append(self.activation_factory())
            if use_dropout:
                model.append(self.dropout_factory())
            model.append(self.conv_factory(in_channels=self.layers[i], out_channels=self.layers[i + 1]))
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor):
        return self.model.forward(x)
