import torch


class BaseObservationModel(torch.nn.Module):
    def __init__(
        self,
        additional_noise_std: float,
        random_mask_fraction: float,
        mask_fill_value: float = 0.0,
    ):
        super().__init__()
        self.additional_noise_std = additional_noise_std
        self.random_mask_fraction = random_mask_fraction
        self.mask_fill_value = mask_fill_value

    def apply_additional_noise(self, data: torch.Tensor) -> torch.Tensor:
        size = data.size()
        noise = torch.normal(mean=0, std=self.additional_noise_std, size=size, device="cpu").to(data.device)
        return data + noise

    def apply_random_mask(self, data: torch.Tensor, dim=-1) -> (torch.Tensor, torch.Tensor):
        size = data.size()
        n_masked_per_step = int(size[dim] * self.random_mask_fraction)
        sample = torch.rand(size, device="cpu").topk(n_masked_per_step, dim=dim).indices.to(data.device)
        mask = torch.zeros(size, device="cpu", dtype=torch.bool).to(data.device)
        mask.scatter_(dim=dim, index=sample, value=True)
        data = torch.masked_fill(data, mask, value=self.mask_fill_value)
        mask_inverse = torch.logical_not(mask).float()
        return data, mask_inverse


class RandomObservationModel(BaseObservationModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """ Random observation model
        Corrupting the data with additional noise sampled from gaussian distribution.
        Some datapoints per time step are randomly masked.
        Args:
            additional_noise_std (float): standard deviation of additional noise.
            random_mask_fraction (float): fraction of random mask.
            mask_fill_value (float): value to fill masked locations. default: 0.0
        """

    def forward(self, data: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        data = self.apply_additional_noise(data)
        data, mask_inverse = self.apply_random_mask(data, dim=-1)
        return data, mask_inverse


class EvenLocationsObservationModel(BaseObservationModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """ Even locations observation model
        Corrupting the data with additional noise sampled from gaussian distribution.
        Masking even locations.
        Args:
            additional_noise_std (float): standard deviation of additional noise.
            random_mask_fraction (float): fraction of random mask.
            mask_fill_value (float): value to fill masked locations. default: 0.0
        """

    def forward(self, data: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        data = self.apply_additional_noise(data)
        data, mask_inverse = self.apply_random_mask(data, dim=-2)
        mask_inverse[:, ::2] = 0.0
        data[..., ::2] = self.mask_fill_value
        return data, mask_inverse


class RandomLocationsObservationModel(BaseObservationModel):
    def __init__(self, location_mask_fraction: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """ Random locations observation model
        Corrupting the data with additional noise sampled from gaussian distribution.
        Masking some random locations.
        Args:
            additional_noise_std (float): standard deviation of additional noise.
            random_mask_fraction (float): fraction of random mask.
            location_mask_fraction (float): fraction of random locations to mask.
            mask_fill_value (float): value to fill masked locations. default: 0.0
        """
        self.location_mask_fraction = location_mask_fraction

    def forward(self, data: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        data = self.apply_additional_noise(data)
        data, mask_inverse = self.apply_random_mask(data, dim=-2)
        size = data.size()
        n_loc_masked = int(size[-1] * self.location_mask_fraction)
        loc_indexes = torch.randperm(size[-1])[:n_loc_masked]
        mask_inverse[..., loc_indexes] = 0.0
        data[..., loc_indexes] = self.mask_fill_value
        return data, mask_inverse
