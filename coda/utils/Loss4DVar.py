from typing import Union

import torch


class Four4DVarLoss:
    """4DVar loss function.

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
                list containing observations and estimated ICs tensor [Batch, 1, Space]
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
            model_loss = self.calculate_model_loss(prediction[1], target[1])
            output["ModelLoss"] = model_loss.detach().clone()
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
        target: torch.Tensor,
    ):
        alpha = self.alpha
        if alpha is None:
            alpha = 1 / (prediction - target).var()
        return torch.nn.functional.mse_loss(prediction, target) * alpha
