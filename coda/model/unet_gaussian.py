from typing import Optional

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

from coda.model.fully_convolutional import FullyConvolutionalNetwork


class ConvolutionalEncodingBlock(FullyConvolutionalNetwork):
    def __init__(self, pooling: Optional[DictConfig] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pooling_cfg = pooling
        if self.pooling_cfg:
            self.pooling = hydra.utils.instantiate(self.pooling_cfg)
        self.x_buffer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        self.x_buffer = x
        if self.pooling_cfg:
            x = self.pooling(x)
        return x


class ConvolutionalDecodingBlock(FullyConvolutionalNetwork):
    def __init__(self, upscale: Optional[DictConfig] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upscale_cfg = upscale
        if self.upscale_cfg:
            # crutches to override in_channels and out_channels if they exist in config
            if hasattr(self.upscale_cfg, "in_channels"):
                self.upscale_cfg.in_channels = int(self.layers[0] / 2)
            if hasattr(self.upscale_cfg, "out_channels"):
                self.upscale_cfg.out_channels = int(self.layers[0] / 2)
            self.upscale = hydra.utils.instantiate(self.upscale_cfg)

    def forward(self, x: torch.Tensor, x_buffer: torch.Tensor = None) -> torch.Tensor:
        if self.upscale_cfg:
            x = self.upscale.forward(x)
            x = torch.concat((x, x_buffer), dim=1)
        x = super().forward(x)
        return x


class GlobalMaxPool(nn.Module):
    def __init__(self, dim: list[int]):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.amax(x, dim=self.dim)


class GlobalAvgPool(nn.Module):
    def __init__(self, dim: list[int]):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, dim=self.dim)


def encoder_builder(
    levels: list[list[int]],
    block: DictConfig,
):
    block_factory = hydra.utils.instantiate(block, _partial_=True)
    blocks = nn.ModuleList()
    n_levels = len(levels)
    for level, level_layers in enumerate(levels):
        if level == n_levels - 1:
            block = block_factory(layers=level_layers, pooling=None)
        else:
            block = block_factory(layers=level_layers)
        blocks.append(block)
    return blocks


def decoder_builder(
    levels: list[list[int]],
    block: DictConfig,
):
    block_factory = hydra.utils.instantiate(block, _partial_=True)
    blocks = nn.ModuleList()
    for level, level_layers in enumerate(levels):
        if level == 0:
            block = block_factory(layers=level_layers, upscale=None)
        else:
            block = block_factory(layers=level_layers)
        blocks.append(block)
    return blocks


class Unet(nn.Module):
    def __init__(
        self,
        encoder: DictConfig,  # encoder builder function
        decoder: DictConfig,  # decoder builder function
        output_convolution: DictConfig,
        output_convolution_sigma: DictConfig,
        #sigma_final_nonlin: DictConfig={"nonlin": "softplus"},
        global_pool: DictConfig = None,
    ):
        super().__init__()
        self.encoder = hydra.utils.call(encoder)
        self.decoder = hydra.utils.call(decoder)

        self.encoder_len = len(self.encoder)
        self.decoder_len = len(self.decoder)
        if self.encoder_len != self.decoder_len:
            raise ValueError(f"Decoder and Encoder length do not match: {self.encoder_len} != {self.decoder_len}")

        self.output_convolution = hydra.utils.instantiate(output_convolution)
        self.output_convolution_sigma = hydra.utils.instantiate(output_convolution_sigma)
        #if sigma_final_nonlin.nonlin == "softplus":
            #self.final_sigma_nonlin = nn.Softplus()
        #else:
            #self.final_sigma_nonlin = nn.Softplus()
        self.final_sigma_nonlin = nn.Softplus()
        #print(sigma_final_nonlin.nonlin, self.final_sigma_nonlin)
        if global_pool:
            self.global_pool = hydra.utils.instantiate(global_pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.encoder:
            x = block.forward(x)
        if hasattr(self, "global_pool"):
            x = self.global_pool(x)
        x = self.decoder[0].forward(x)
        for i in range(1, self.decoder_len):
            j = self.decoder_len - 1 - i
            x_buffer = self.encoder[j].x_buffer
            if hasattr(self, "global_pool"):
                x_buffer = self.global_pool(x_buffer)
            x = self.decoder[i].forward(x, x_buffer)

        #x = self.output_convolution.forward(x)
        #return x
        mu = self.output_convolution.forward(x)
        sigma = self.final_sigma_nonlin(self.output_convolution_sigma.forward(x))
        return mu, sigma
        
