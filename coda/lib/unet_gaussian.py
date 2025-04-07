from dataclasses import dataclass
from typing import Any
import torch.nn


@dataclass
class GlobalMaxPool:
    _target_: str = "coda.model.unet.GlobalMaxPool"
    dim: Any = -1


@dataclass
class GlobalAvgPool:
    _target_: str = "coda.model.unet.GlobalAvgPool"
    dim: Any = -1


@dataclass
class ConvolutionalEncodingBlock:
    _target_: str = "coda.model.unet.ConvolutionalEncodingBlock"
    _recursive_: bool = False
    convolution: Any = None
    activation: Any = None
    layers: Any = None
    pooling: Any = None
    batch_norm: Any = None
    dropout: Any = None


@dataclass
class ConvolutionalDecodingBlock:
    _target_: str = "coda.model.unet.ConvolutionalDecodingBlock"
    _recursive_: bool = False
    convolution: Any = None
    activation: Any = None
    layers: Any = None
    upscale: Any = None
    batch_norm: Any = None
    dropout: Any = None


@dataclass
class ConvolutionalEncoder:
    _target_: str = "coda.model.unet.encoder_builder"
    _recursive_: bool = False
    levels: Any = None
    block: Any = None


@dataclass
class ConvolutionalDecoder:
    _target_: str = "coda.model.unet.decoder_builder"
    _recursive_: bool = False
    levels: Any = None
    block: Any = None


@dataclass
class Unet_gaussian:
    _target_: str = "coda.model.unet.Unet"
    _recursive_: bool = False
    encoder: Any = None
    decoder: Any = None
    output_convolution: Any = None
    output_convolution_sigma: Any = None
    global_pool: Any = None
    #sigma_final_nonlin: Any = None