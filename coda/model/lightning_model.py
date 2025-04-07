import os
from typing import Any

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from mdml_tools.simulators.base import BaseSimulator
from mdml_tools.utils.logging import get_logger
from omegaconf import DictConfig
from torchmetrics import MeanSquaredError


class LightningBaseModel(pl.LightningModule):
    """Base Lightning Module. This module shares common functionality for three tasks:

    - Data Assimilation: training a deep data assimilation network
    - Parameter Tuning: training a deep data assimilation network and fitting free model parameters
    - Parametrization Learning: training parametrization along deep data assimilation network
    """

    def __init__(
        self,
        simulator: DictConfig,
        assimilation_network: DictConfig,
        loss: DictConfig,
        rollout_length: int,
        time_step: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.console_logger = get_logger(__name__)
        self.console_logger.info(f"Instantiating simulator <{simulator._target_}>")
        self.simulator: BaseSimulator = hydra.utils.instantiate(simulator)
        self.console_logger.info(f"Instantiating assimilation network <{assimilation_network._target_}>")
        self.assimilation_network: nn.Module = hydra.utils.instantiate(assimilation_network)
        self.console_logger.info(f"Instantiating loss function <{loss._target_}>")
        self.loss_function = hydra.utils.instantiate(loss)
        self.rollout_length = rollout_length + 1
        self.time_step = time_step
        self.rmse_func = MeanSquaredError(False)

    def rollout(self, ic: torch.Tensor) -> torch.Tensor:
        if isinstance(self.simulator, BaseSimulator):
            t = torch.arange(0, self.rollout_length * self.time_step, self.time_step)
            return self.simulator.integrate(t, ic)
        else:
            raise NotImplementedError("The simulator should be child of BaseSimulator class")

    def do_step(self, batch: torch.Tensor, stage: str = "Training") -> torch.Tensor:
        # Training variables
        observations_data = batch[0]
        observations_mask = batch[1]
        feed_forward_left = batch[2]
        feed_forward_right = batch[3]

        # Variables to compute metrics
        true_state_left = batch[5]
        true_state_right = batch[6]

        estimated_state_left = self.assimilation_network.forward(feed_forward_left)
        estimated_state_right = self.assimilation_network.forward(feed_forward_right)
        rollout = self.rollout(estimated_state_left)

        if self.loss_function.use_model_term:
            loss_dict: dict = self.loss_function(
                prediction=[rollout, rollout[:, -1, :].unsqueeze(1)],
                target=[observations_data, estimated_state_right],
                mask=observations_mask,
            )
        else:
            loss_dict: dict = self.loss_function(rollout, observations_data, observations_mask)

        for key, value in loss_dict.items():
            if value is not None:
                self.log(f"{key}/{stage}", value)

        with torch.no_grad():
            rmse_state_left = self.rmse_func(estimated_state_left.squeeze(1), true_state_left)
            rmse_state_right = self.rmse_func(estimated_state_right.squeeze(1), true_state_right)
            self.log(f"RMSEStateLeft/{stage}", rmse_state_left)
            self.log(f"RMSEStateRight/{stage}", rmse_state_right)

        # Log metrics on validation step
        if stage == "Validation":
            self.log("hp/data_missmatch", loss_dict["DataLoss"])
            if "ModelLoss" in loss_dict and loss_dict["ModelLoss"] is not None:
                self.log("hp/model_error", loss_dict["ModelLoss"])
            self.log("hp/ic_error_left", rmse_state_left)
            self.log("hp/ic_error_right", rmse_state_right)

        return loss_dict["TotalLoss"]

    def on_save_checkpoint(self, *args, **kwargs):
        chekpoints_dir = self.trainer.checkpoint_callback.dirpath
        if not os.path.exists(chekpoints_dir):
            os.makedirs(chekpoints_dir)
        torch.save(self.assimilation_network, os.path.join(chekpoints_dir, "assimilation_network.ckpt"))
        simulator_params = sum([torch.prod(torch.tensor(p.size())).item() for p in self.simulator.parameters()])
        if simulator_params > 0:
            torch.save(self.simulator, os.path.join(chekpoints_dir, "simulator.ckpt"))

    def forward(self, input_window: torch.Tensor) -> torch.Tensor:
        return self.assimilation_network.forward(input_window)


class DataAssimilationModule(LightningBaseModel):
    def __init__(
        self,
        simulator: DictConfig,
        assimilation_network: DictConfig,
        optimizer: DictConfig,
        loss: DictConfig,
        rollout_length: int,
        time_step: int,
    ):
        super().__init__(simulator, assimilation_network, loss, rollout_length, time_step)
        self.cfg_optimizer: DictConfig = optimizer.data_assimilation

    def configure_optimizers(self) -> Any:
        params = [*self.assimilation_network.parameters()]
        optimizer = hydra.utils.instantiate(self.cfg_optimizer, params=params)
        return optimizer

    def training_step(self, batch, **kwargs):
        return self.do_step(batch, "Training")

    def validation_step(self, batch, *args, **kwargs):
        return self.do_step(batch, "Validation")


class ParameterTuningModule(LightningBaseModel):
    def __init__(
        self,
        simulator: DictConfig,
        assimilation_network: DictConfig,
        optimizer: DictConfig,
        loss: DictConfig,
        rollout_length: int,
        time_step: int,
    ):
        super().__init__(simulator, assimilation_network, loss, rollout_length, time_step)
        self.cfg_optimizer_da: DictConfig = optimizer.data_assimilation
        self.cfg_optimizer_param: DictConfig = optimizer.parametrization
        self.automatic_optimization = False
        self.simulator.forcing = None  # we should allow none for the simulator
        if self.simulator.forcing is None:
            self.simulator.forcing = nn.Parameter(
                data=torch.randint(low=4, high=14, size=(1,), dtype=torch.float64),
                requires_grad=True,
            )

    def configure_optimizers(self) -> Any:
        params_data_assimilation = [*self.assimilation_network.parameters()]
        optimizer_da = hydra.utils.instantiate(self.cfg_optimizer_da, params=params_data_assimilation)
        params_simulator_parameters = [*self.simulator.parameters()]
        optimizer_param = hydra.utils.instantiate(self.cfg_optimizer_param, params=params_simulator_parameters)
        return optimizer_da, optimizer_param

    def training_step(self, batch, **kwargs):
        optimizers = self.optimizers()
        loss = self.do_step(batch, "Training")
        for optimizer in optimizers:
            optimizer.zero_grad()
        self.manual_backward(loss)
        for optimizer in optimizers:
            optimizer.step()
        self.log("Parameter/Training", self.simulator.forcing)

    def validation_step(self, batch, *args, **kwargs):
        self.do_step(batch, "Validation")
        self.log("Parameter/Validation", self.simulator.forcing)


class ParametrizationLearningModule(LightningBaseModel):
    def __init__(
        self,
        simulator: DictConfig,
        assimilation_network: DictConfig,
        optimizer: DictConfig,
        loss: DictConfig,
        rollout_length: int,
        time_step: int,
    ):
        super().__init__(simulator, assimilation_network, loss, rollout_length, time_step)
        self.cfg_optimizer_da: DictConfig = optimizer.data_assimilation
        self.cfg_optimizer_param: DictConfig = optimizer.parametrization
        self.automatic_optimization = False

    def configure_optimizers(self) -> Any:
        params_data_assimilation = [*self.assimilation_network.parameters()]
        optimizer_da = hydra.utils.instantiate(self.cfg_optimizer_da, params=params_data_assimilation)
        params_parametrization = [*self.simulator.parameters()]
        optimizer_param = hydra.utils.instantiate(self.cfg_optimizer_param, params=params_parametrization)
        return optimizer_da, optimizer_param

    def training_step(self, batch, **kwargs):
        optimizers = self.optimizers()
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss = self.do_step(batch, "Training")
        self.manual_backward(loss)

        for optimizer in optimizers:
            optimizer.step()

    def validation_step(self, batch, *args, **kwargs):
        self.do_step(batch, "Validation")
