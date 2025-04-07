import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.random
from mdml_tools.utils import logging as mdml_logging
from omegaconf import DictConfig

from coda.initializations import init_callbacks, init_datamodule, init_lightning_module, init_logger, init_trainer
from coda.lib.config import register_configs

register_configs()


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig):
    if cfg.print_config:
        mdml_logging.print_config(cfg, resolve=True)

    console_logger = mdml_logging.get_logger(__name__)
    console_logger.info(f"Hydra version: {hydra.__version__}")

    console_logger.info(f"Applying random seed: {cfg.random_seed}")
    torch.random.manual_seed(cfg.random_seed)
    pl.seed_everything(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    try:
        logger = init_logger(cfg, console_logger)
        callbacks = init_callbacks(cfg, console_logger)
        trainer = init_trainer(cfg, logger, callbacks, console_logger)

        lightning_module = init_lightning_module(cfg, console_logger)

        # load assimilation network from checkpoint if provided (used for parametrization learning)
        if cfg.assimilation_network_checkpoint:
            console_logger.info("Loading assimilation network from checkpoint")
            lightning_module.assimilation_network = torch.load(cfg.assimilation_network_checkpoint, map_location="cpu")

        hydra_params = mdml_logging.get_hparams_from_hydra_config(config=cfg, model=lightning_module)

        # delete all dicts and lists from hydra_params
        keys_to_delete = []
        for key, value in hydra_params.items():
            if isinstance(value, DictConfig) or isinstance(value, list) or isinstance(value, dict):
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del hydra_params[key]

        for this_logger in logger:
            if "tensorboard" in str(this_logger):
                console_logger.info("Add hparams to tensorboard")
                this_logger.log_hyperparams(
                    hydra_params,
                    {
                        "hp/data_missmatch": 0,
                        "hp/model_error": 0,
                        "hp/ic_error_left": 0,
                        "hp/ic_error_right": 0,
                    },
                )
            else:
                this_logger.log_hyperparams(hydra_params)

        datamodule = init_datamodule(cfg, console_logger)
        if cfg.training:
            console_logger.info("Training...")
            trainer.fit(model=lightning_module, datamodule=datamodule)
            console_logger.info("Training finished")

    except Exception as exception:
        console_logger.exception(f"Error occurred during main(): {exception}.")
        print("Error!", exception)
        raise exception


if __name__ == "__main__":
    main()
