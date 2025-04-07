from logging import Logger

import hydra
from mdml_tools.utils.logging import set_tb_logger
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger as LightningLogger


def init_callbacks(cfg: DictConfig, console_logger: Logger = None) -> list[Callback]:
    """Init Lightning callbacks.

    Args:
        cfg (DictConfig): The configuration object from hydra.
        console_logger (Logger): Python logging instance to log to a console.

    Returns:
        list[Callback]: The list of initialized callbacks.
    """
    callbacks: list[Callback] = []
    if "lightning_callback" in cfg and cfg["lightning_callback"] is not None:
        for _, cb_conf in cfg["lightning_callback"].items():
            if "_target_" in cb_conf:
                if console_logger:
                    console_logger.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def init_logger(cfg: DictConfig, console_logger: Logger = None) -> list[LightningLogger]:
    """Init Lightning loggers (Tensorboard, WanDB, ...).

    Args:
        cfg (DictConfig): The configuration object from hydra.
        console_logger (Logger): Python logging instance to log to a console.

    Returns:
        list[LightningLoggerBase]: The list of initialized loggers.
    """
    logger: list[LightningLogger] = []
    if "lightning_logger" in cfg:
        for _, lg_conf in cfg["lightning_logger"].items():
            if "_target_" in lg_conf:
                console_logger.info(f"Instantiating logger <{lg_conf._target_}>")
                logger_instance = hydra.utils.instantiate(lg_conf)
                logger.append(logger_instance)
                if "tensorboard" in lg_conf._target_.lower():
                    set_tb_logger(logger_instance)
    return logger


def init_trainer(
    cfg: DictConfig, logger: list[LightningLogger], callbacks: list[Callback], console_logger: Logger = None
) -> Trainer:
    """Init the Lightning lightning_trainer.

    Args:
        cfg (DictConfig): The configuration object from hydra.
        logger (list[LightningLoggerBase]): The initialized loggers.
        callbacks (list[Callback]): The initialized callbacks.
        console_logger (Logger): Python logging instance to log to a console.

    Returns:
        Trainer: The initialized lightning_trainer.
    """
    if console_logger:
        console_logger.info(f"Instantiating lightning_trainer <{cfg.lightning_trainer._target_}>")
    if len(callbacks) == 0:
        callbacks = None

    trainer: Trainer = hydra.utils.instantiate(
        cfg.lightning_trainer, logger=logger, callbacks=callbacks, _convert_="partial"
    )
    return trainer


def init_lightning_module(cfg: DictConfig, console_logger: Logger = None) -> LightningModule:
    """Init the lightning module.

    Args:
        cfg (DictConfig): The configuration object from hydra.
        console_logger (Logger): Python logging instance to log to a console.

    Returns:
        LightningModule: The initialized pytoch_lightning module.
    """
    if console_logger:
        console_logger.info(f"Initializing lightning module <{cfg.lightning_module._target_}>")
    lightning_module: LightningModule = hydra.utils.instantiate(
        cfg.lightning_module,
        simulator=cfg.simulator,
        assimilation_network=cfg.assimilation_network,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
    )
    return lightning_module


def init_datamodule(cfg: DictConfig, console_logger: Logger = None) -> LightningDataModule:
    """Init the lightning datamodule.

    Args:
        cfg (DictConfig): The configuration object from hydra.
        console_logger (Logger): Python logging instance to log to a console.

    Returns:
        LightningDataModule: The initialized pytoch_lightning datamodule.
    """
    if console_logger:
        console_logger.info(f"Initializing lightning module <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    return datamodule
