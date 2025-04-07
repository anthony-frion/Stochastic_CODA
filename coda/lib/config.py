from dataclasses import dataclass
from typing import Any

from hydra.conf import ConfigStore, MISSING
from mdml_tools.utils import add_hydra_models_to_config_store

from coda.lib.datamodule import L96DataLoader, L96InferenceDataset, L96TrainingDataset
from coda.lib.lightning_module import (
    DataAssimilationModule, 
    DataAssimilationGaussianModule, 
    DataAssimilationGaussianLRModule,
    ParameterTuningModule, 
    ParametrizationLearningModule
)
from coda.lib.loss import Four4DVarLoss, NegativeLogLikelihoodLoss, NegativeLogLikelihoodLoss_Full
from coda.lib.model import FullyConvolutionalNetwork, L96Parametrized
from coda.lib.observation_models import (
    EvenLocationsObservationModel,
    RandomLocationsObservationModel,
    RandomObservationModel,
)

from coda.lib.unet import (
    ConvolutionalDecoder,
    ConvolutionalDecodingBlock,
    ConvolutionalEncoder,
    ConvolutionalEncodingBlock,
    GlobalAvgPool,
    GlobalMaxPool,
    Unet,
)

from coda.lib.unet_gaussian import (
    Unet_gaussian
)


def register_configs() -> None:
    cs = ConfigStore.instance()
    # add hydra models from mdml_tools
    add_hydra_models_to_config_store(cs)

    # loss:
    cs.store(name="4dvar", node=Four4DVarLoss, group="loss")
    cs.store(name="NLL", node=NegativeLogLikelihoodLoss, group="loss")
    cs.store(name="NLL_full", node=NegativeLogLikelihoodLoss_Full, group="loss")

    # simulator:
    model_group = "simulator"
    cs.store(name="l96_parametrized_base", node=L96Parametrized, group=model_group)

    # assimilation network
    cs.store("unet_base", node=Unet, group="assimilation_network")
    print('pouet')
    cs.store("unet_gaussian", node=Unet_gaussian, group="assimilation_network")
    print("pouet pouet")
    cs.store("conv_encoder_base", node=ConvolutionalEncoder, group="assimilation_network/encoder")
    cs.store("conv_block_encoding_base", node=ConvolutionalEncodingBlock, group="assimilation_network/encoder/block")
    cs.store("conv_decoder_base", node=ConvolutionalDecoder, group="assimilation_network/decoder")
    cs.store("conv_block_decoding_base", node=ConvolutionalDecodingBlock, group="assimilation_network/decoder/block")
    cs.store("global_max_pool_base", node=GlobalMaxPool, group="assimilation_network/global_pool")
    cs.store("global_avg_pool_base", node=GlobalAvgPool, group="assimilation_network/global_pool")

    # parametrization:`
    cs.store(name="fully_convolutional_network_base", node=FullyConvolutionalNetwork, group="simulator/parametrization")

    # lightning module:
    cs.store(name="data_assimilation_module", node=DataAssimilationModule, group="lightning_module")
    cs.store(name="data_assimilation_gaussian_module", node=DataAssimilationGaussianModule, group="lightning_module")
    cs.store(name="data_assimilation_gaussian_LR_module", node=DataAssimilationGaussianLRModule, group="lightning_module")
    cs.store(name="parameter_tuning_module", node=ParameterTuningModule, group="lightning_module")
    cs.store(name="parametrization_learning_module", node=ParametrizationLearningModule, group="lightning_module")

    # datamodule:
    cs.store(name="l96_datamodule_base", node=L96DataLoader, group="datamodule")
    cs.store(name="l96_training_dataset_base", node=L96TrainingDataset, group="datamodule/dataset")
    cs.store(name="l96_inference_dataset_base", node=L96InferenceDataset, group="datamodule/dataset")
    cs.store(name="random_observation_model_base", node=RandomObservationModel, group="datamodule/observation_model")
    cs.store(
        name="even_locations_observation_model_base",
        node=EvenLocationsObservationModel,
        group="datamodule/observation_model",
    )
    cs.store(
        name="random_locations_observation_model_base",
        node=RandomLocationsObservationModel,
        group="datamodule/observation_model",
    )

    # register the base config class (this name has to be called in config.yaml):
    cs.store(name="base_config", node=Config)
    print("pouet pouet pouet")


@dataclass
class Config:
    output_dir_base_path: str = MISSING
    print_config: bool = True
    training: bool = True
    random_seed: int = 101
    debug: bool = False
    time_step: float = MISSING
    rollout_length: int = MISSING
    input_window_extend: int = MISSING
    batch_size: int = MISSING
    l96_forcing: float = MISSING
    loss_alpha: float = MISSING

    simulator: Any = MISSING
    assimilation_network: Any = MISSING
    loss: Any = MISSING
    lightning_module: Any = MISSING
    assimilation_network_checkpoint: Any = None
    optimizer: Any = MISSING

    datamodule: Any = MISSING
    lightning_trainer: Any = MISSING
    lightning_logger: Any = MISSING
    lightning_callback: Any = None
