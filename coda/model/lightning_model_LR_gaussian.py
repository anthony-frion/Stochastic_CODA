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

from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal


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
        sample_mean: bool,
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
        self.sample_mean = sample_mean
        print(f"Lightning model sample mean: {self.sample_mean}")
        self.rmse_func = MeanSquaredError(False)

    def rollout(self, ic: torch.Tensor) -> torch.Tensor:
        if isinstance(self.simulator, BaseSimulator):
            t = torch.arange(0, self.rollout_length * self.time_step, self.time_step)
            return self.simulator.integrate(t, ic)
        else:
            raise NotImplementedError("The simulator should be child of BaseSimulator class")

    def do_step(self, batch: torch.Tensor, stage: str = "Training") -> torch.Tensor:
        # Training variables
        observations_data = batch[0] # [Batch_size, Rollout_time+1, Space]
        observations_mask = batch[1] # [Batch_size, Rollout_time+1, Space]
        feed_forward_left = batch[2] # [Batch_size, 2*Rollout_time+1, Space]
        feed_forward_right = batch[3] # [Batch_size, 2*Rollout_time+1, Space]

        # Variables to compute metrics
        true_state_left = batch[5] # [Batch_size, Space]
        true_state_right = batch[6] # [Batch_size, Space]

        mu_left, Sigma_diag_left, Sigma_factor_left = self.assimilation_network.forward(feed_forward_left)
        mu_right, Sigma_diag_right, Sigma_factor_right = self.assimilation_network.forward(feed_forward_right)
        if self.sample_mean:
            #print("WTFFFFFFFF")
            estimated_state_left = mu_left
            estimated_state_right = mu_right
        else:
            #print('shapes of inputs to the sampling:')
            #print(mu_left.shape, Sigma_factor_left.shape, Sigma_diag_left.shape)
            estimated_state_left = LowRankMultivariateNormal(mu_left.squeeze(1), Sigma_factor_left, Sigma_diag_left).rsample()
            estimated_state_right = LowRankMultivariateNormal(mu_right.squeeze(1), Sigma_factor_right, Sigma_diag_right).rsample()
            #estimated_state_left = sample_from_diag_gaussian(mu_left, Sigma_diag_left.unsqueeze(1))
            #estimated_state_right = sample_from_diag_gaussian(mu_right, Sigma_diag_right.unsqueeze(1))
            #estimated_state_left = LowRankMultivariateNormal(mu_left.squeeze(1), torch.zeros_like(Sigma_factor_left), Sigma_diag_left).rsample()
            #estimated_state_right = LowRankMultivariateNormal(mu_right.squeeze(1), torch.zeros_like(Sigma_factor_right), Sigma_diag_right).rsample()
        rollout = self.rollout(estimated_state_left)
        #print("shapes of inputs to the loss:")
        #print(rollout[:, -1, :].unsqueeze(1).shape, mu_right.shape, Sigma_diag_right.shape, Sigma_factor_right.shape)
        loss_dict: dict = self.loss_function(
            prediction=[rollout, rollout[:, -1, :].unsqueeze(1)],
            target=[observations_data, mu_right, Sigma_diag_right, Sigma_factor_right],
            mask=observations_mask,
        )
        
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
        sample_mean: bool,
    ):
        super().__init__(simulator, assimilation_network, loss, rollout_length, time_step, sample_mean)
        self.cfg_optimizer: DictConfig = optimizer.data_assimilation

    def configure_optimizers(self) -> Any:
        params = [*self.assimilation_network.parameters()]
        optimizer = hydra.utils.instantiate(self.cfg_optimizer, params=params)
        return optimizer

    def training_step(self, batch, **kwargs):
        return self.do_step(batch, "Training")

    def validation_step(self, batch, *args, **kwargs):
        return self.do_step(batch, "Validation")

def sample_from_diag_gaussian(mu, sigma, n_samples=1, device='cuda', debug=False):
    std = torch.sqrt(sigma)
    eps = torch.randn(n_samples, std.shape[0], std.shape[1], std.shape[2]).to(device)
    if n_samples == 1:
        eps = eps.squeeze(0)
    #if debug:
        #print(mu, std, eps)
        #print(mu.shape, std.shape, eps.shape)
    samples = mu + eps*std
    return samples
