from typing import Union

import numpy as np
import torch
import torch.nn as nn

from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal

class NegativeLogLikelihoodLoss_Full:
    """Negative log likelihood (NLL) loss function.
    We assume that the CODA models outputs a Gaussian distribution.
    The data consistency term is the NLL of the groundtruth regarding the distribution of the predictions.
    The model consistency term is the NLL of the 'mean' prediction
    with regards to the model distribution computed from the time-advanced observations.

    Args:
        alpha (float): simulator error scaler is None by default
            use 1 / model_error_variance if alpha is not provided
    """

    def __init__(
        self,
        alpha: float = None,
    ):
        self.alpha = alpha
        self.use_model_term = True if alpha > 0 else False
        self.device = None

    def __call__(
        self,
        prediction: Union[torch.Tensor, list[torch.Tensor]],
        target: Union[torch.Tensor, list[torch.Tensor]],
        mask: torch.Tensor = None,
    ) -> dict[str : torch.Tensor]:
        """Calculate 4DVar loss function.
        Args:
            prediction (Union[torch.Tensor, list[torch.Tensor]]): rollout tensor [Batch, Time, Space] or
                list containing rollout and predicted ICs tensor [Batch, 1, Space]
            target (Union[torch.Tensor, list[torch.Tensor]]): observations tensor [Batch, Time, Space] or
                list containing observations, 
                    estimated right means [Batch, 1, Space]
                    and estimated right variances [Batch, 1, Space]
            mask (torch.Tensor): boolean observations mask tensor where False is masked value.

        Returns:
             dict[str: torch.Tensor]: dictionary containing loss values;
             if alpha == 0 then contain keys ["DataLoss", "TotalLoss"]
             else contain keys ["DataLoss", "ModelLoss", "TotalLoss"]
        """
        if isinstance(prediction, torch.Tensor):
            prediction = [prediction]
        if isinstance(target, torch.Tensor):
            target = [target]
        self.device = prediction[0].device

        output = {
            "DataLoss": None,
            "ModelLoss": None,
            "TotalLoss": None,
        }

        loss = torch.zeros(1, device=self.device)
        loss += self.calculate_data_loss(prediction[0], target[0], mask)
        output["DataLoss"] = loss.detach().clone()
        if self.use_model_term:
            model_loss = self.calculate_model_loss(prediction[1], target[1], target[2], target[3])
            output["ModelLoss"] = model_loss.detach().clone()
            #print("shape of model loss:")
            #print(model_loss.shape)
            loss += model_loss
        output["TotalLoss"] = loss
        return output
    
    @staticmethod
    def calculate_data_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        if mask is not None:
            prediction = torch.masked_select(prediction, mask)
            target = torch.masked_select(target, mask)
        return torch.nn.functional.mse_loss(prediction, target)
    
    def calculate_model_loss(
        self,
        prediction: torch.Tensor,
        mu: torch.Tensor,
        Sigma_diag: torch.Tensor,
        Sigma_factor: torch.Tensor
    ):
        alpha = self.alpha
        if alpha is None:
            alpha = 1 / (mu - prediction).var()
         #print("shapes of inputs to lowrankmultivaritenormal:")
         #print(mu.shape, prediction.shape, Sigma_diag.shape, Sigma_factor.shape)
        
        Sigma = torch.diag_embed(Sigma_diag).squeeze(1) + torch.bmm(Sigma_factor, Sigma_factor.permute((0,2,1)))
        
        #print('Values to compare:')
        #print(batched_NLL_full(mu, Sigma, prediction) * alpha)
        #print(torch.mean(- LowRankMultivariateNormal(mu.squeeze(1), Sigma_factor, Sigma_diag).log_prob(prediction.squeeze(1))) * alpha / 40)
        return batched_NLL_full(mu, Sigma, prediction) * alpha
        #print(mu.shape, Sigma_diag.shape)
        #Sigma_diag_ = torch.diagonal(Sigma, dim1=1, dim2=2)
        #return (batched_NLL_full(mu, Sigma, prediction) + batched_NLL_diag(mu, Sigma_diag_.unsqueeze(1), prediction)) * alpha / 2
        #return - LowRankMultivariateNormal(mu.squeeze(1), Sigma_factor, Sigma_diag).log_prob(prediction.squeeze(1))
        #return torch.mean(- LowRankMultivariateNormal(mu.squeeze(1), Sigma_factor, Sigma_diag).log_prob(prediction.squeeze(1))) * alpha / 40

    @staticmethod
    def calculate_data_loss_diag(
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        if mask is not None:
            mu = torch.masked_select(mu, mask)
            sigma = torch.masked_select(sigma, mask)
            target = torch.masked_select(target, mask)
        return batched_NLL_diag(mu, sigma, target)

    def calculate_model_loss_diag(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor
    ):
        alpha = self.alpha
        if alpha is None:
            alpha = 1 / (mu - target).var()
        return batched_NLL_diag(mu, sigma, target) * alpha

def batched_NLL_diag(mu, sigma, target, device='cuda'):
    """Calculate batched negative log likelihoods of Gaussians, with diagonal variance matrices.
    Args:
        mu (torch.Tensor): means of Gaussian distributions [Batch, Time, Space]
        sigma (torch.Tensor): Variance (not std) vectors of Gaussian distributions [Batch, Time, Space]
        target (torch.Tensor): target tensor [Batch, Time, Space]
        mask (torch.Tensor): boolean observations mask tensor where False is masked value.
    
    Returns:
         torch.Tensor: mean of NLL over all batched samples
    """
    #return torch.mean( np.log(2 * torch.pi)/2 + torch.log(sigma)/2 + ((target - mu) **2) / (2 * sigma) )
    return nn.GaussianNLLLoss(full=True)(mu, target, sigma)

def batched_NLL_full(mu, Sigma, target, device='cuda'):
    """Calculate batched negative log likelihoods of Gaussians, when variance matrices are supposedly full.
    There is no time dimension per se but it can be fused with the batch dimension if needed.
    Args:
        mu (torch.Tensor): means of Gaussian distributions [Batch, Space]
        Sigma (torch.Tensor): Variance matrices of Gaussian distributions [Batch, Space, Space]
        target (torch.Tensor): target tensor [Batch, Space]
        mask (torch.Tensor): boolean observations mask tensor where False is masked value.
    
    Returns:
         torch.Tensor: mean of NLL over all batched samples
    """
    n = Sigma.shape[1]
    n_points = Sigma.shape[0]
    if len(target.shape) == 1:
        return - np.log(2 * torch.pi) * n/2 - torch.logdet(Sigma)/2 - \
            torch.matmul(torch.matmul((target - mu), torch.linalg.inv(Sigma)), (target - mu)) / 2
    else:
        Sigma += torch.eye(n, device=device)*1e-9
        left_prod = torch.matmul((target - mu), torch.linalg.inv(Sigma))
        full_prod = torch.matmul(left_prod, (target - mu).permute(0,2,1)).reshape((n_points))
        return torch.mean(np.log(2 * torch.pi) * n/2 + torch.logdet(Sigma)/2 + full_prod / 2) / n
    











