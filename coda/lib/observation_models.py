from dataclasses import dataclass


@dataclass
class RandomObservationModel:
    _target_: str = "coda.datamodule.observational_models.RandomObservationModel"
    _recursive_: bool = False
    additional_noise_std: float = 1.0
    random_mask_fraction: float = 0.75
    mask_fill_value: float = 0.0


@dataclass
class EvenLocationsObservationModel:
    _target_: str = "coda.datamodule.observational_models.EvenLocationsObservationModel"
    _recursive_: bool = False
    additional_noise_std: float = 1.0
    random_mask_fraction: float = 0.5
    mask_fill_value: float = 0.0


@dataclass
class RandomLocationsObservationModel:
    _target_: str = "coda.datamodule.observational_models.RandomLocationsObservationModel"
    _recursive_: bool = False
    additional_noise_std: float = 1.0
    random_mask_fraction: float = 0.5
    location_mask_fraction: float = 0.5
    mask_fill_value: float = 0.0
