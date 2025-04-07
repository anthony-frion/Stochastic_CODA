import os

import h5py
import hydra.utils
import pytorch_lightning as pl
import torch
import torch.nn.functional as tnf
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


class L96BaseDataset(Dataset):
    """Base class for L96 datasets. Corrupting the ground truth data with additional noise and masking.

    Args:
        ground_truth_data (torch.Tensor): ground truth data. This tensor should have shape [1, time, x_grid_size]
        observation_model (torch.nn.Module): observation model configuration
        path_to_save_data (str): path to save observations data. default: None
    """

    def __init__(
        self,
        ground_truth_data: torch.Tensor,
        observation_model: torch.nn.Module,
        path_to_save_data: str = None,
    ):
        self.ground_truth = ground_truth_data
        self.observation_model = observation_model
        self.observations, self.mask = self.observation_model.forward(self.ground_truth)

        if path_to_save_data is not None:
            if not os.path.exists(path_to_save_data):
                os.makedirs(path_to_save_data)
            torch.save(self.ground_truth, os.path.join(path_to_save_data, "ground_truth.pt"))
            torch.save(self.observations, os.path.join(path_to_save_data, "observations.pt"))
            torch.save(self.mask, os.path.join(path_to_save_data, "mask.pt"))


class L96TrainingDataset(L96BaseDataset):
    """Data assimilation related training dataset for L96. This dataset can be used to train data assimilation model,
    tune model parameters along data assimilation, and train a parametrization through data assimilation procedure.

    - It uses only one trajectory of the L96 model.
    - Edge samples where observations are not available are ignored.

    Args:
        rollout_length (int): number of steps through forward operator.
        input_window_extend (int): half-length of neural network input. default: rollout_length
        extend_channels (bool): whether to add additional channels to input tensor. default: True
    """

    def __init__(
        self,
        rollout_length: int,
        input_window_extend: int = None,
        extend_channels: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rollout_length = rollout_length
        self.window_extend = rollout_length
        self.use_standard_input_window_extend = True
        if input_window_extend is not None:
            self.window_extend = input_window_extend
            self.use_standard_input_window_extend = False

        self.extend_channels = extend_channels

        self.data = torch.concat([self.ground_truth, self.observations, self.mask], dim=0)
        self.n_time_steps = self.data.size(1)
        if self.use_standard_input_window_extend:
            end_point = self.n_time_steps - 3 * self.window_extend - 1
            self.sampling_indexes = torch.arange(self.window_extend, end_point, step=1) + self.window_extend
        else:
            end_point = self.n_time_steps - 2 * self.window_extend - rollout_length - 1
            self.sampling_indexes = torch.arange(self.window_extend, end_point, step=1) + self.window_extend

    def __len__(self):
        return len(self.sampling_indexes)

    def add_channels_to_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add several channels to input tensor."""
        # define expand shape
        expand_shape = tensor.size()
        expand_shape = (1, expand_shape[-2], expand_shape[-1])

        # generate data for new channels
        relative_indexes = torch.arange(-self.window_extend, self.window_extend + 1)
        relative_indexes = relative_indexes.unsqueeze(-1).expand(expand_shape)
        relative_indexes_mul_state = tensor[0, :, :] * relative_indexes
        tensor = torch.cat((tensor, relative_indexes, relative_indexes_mul_state), dim=0)
        return tensor

    def __getitem__(self, index: int):
        start_index = self.sampling_indexes[index]
        end_index = start_index + self.rollout_length + 1

        # variables to calculate missmatch between observations and rollout
        rollout_data = self.data[1, start_index:end_index]
        rollout_mask = self.data[2, start_index:end_index].bool()

        # data assimilation network input
        feed_forward_start = self.data[1:, start_index - self.window_extend : start_index + self.window_extend + 1]
        feed_forward_end = self.data[1:, end_index - self.window_extend : end_index + self.window_extend + 1]

        if self.extend_channels:
            feed_forward_start = self.add_channels_to_tensor(feed_forward_start)
            feed_forward_end = self.add_channels_to_tensor(feed_forward_end)

        # extra variables to calculate metrics during training
        ground_truth_data = self.data[0, start_index:end_index]
        true_state_start = self.data[0, start_index]
        true_state_end = self.data[0, end_index]

        return (
            rollout_data,
            rollout_mask,
            feed_forward_start,
            feed_forward_end,
            ground_truth_data,
            true_state_start,
            true_state_end,
        )


class L96InferenceDataset(L96BaseDataset):
    def __init__(
        self,
        input_window_extend: int,
        observations: torch.Tensor = None,
        mask: torch.Tensor = None,
        extend_channels: bool = True,
        drop_edge_samples: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.window_extend = input_window_extend
        self.extend_channels = extend_channels
        self.drop_edge_samples = drop_edge_samples

        if observations is not None:
            self.observations = observations
        if mask is not None:
            self.mask = mask

        self.data = torch.stack([self.ground_truth, self.observations, self.mask], dim=1)
        self.zeros_padding_data_tensors()
        self.n_time_steps = self.data.size(-2)
        self.sampling_indexes = self.get_sampling_indexes()

    def __len__(self):
        return len(self.sampling_indexes)

    def zeros_padding_data_tensors(self) -> None:
        self.data = self.data.movedim(-2, -1)
        self.data = tnf.pad(self.data, (self.window_extend, self.window_extend + 1), mode="constant", value=0)
        self.data = self.data.movedim(-1, -2)

    def get_sampling_indexes(self) -> torch.Tensor:
        if self.drop_edge_samples:
            end_point = self.n_time_steps - 3 * self.window_extend
            sampling_indexes = torch.arange(self.window_extend, end_point, step=1) + self.window_extend
        else:
            end_point = self.n_time_steps - 2 * self.window_extend - 1
            sampling_indexes = torch.arange(0, end_point, step=1) + self.window_extend
        return sampling_indexes

    def add_channels_to_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add several channels to input tensor."""
        # define expand shape
        expand_shape = tensor.size()
        expand_shape = (expand_shape[0], 1, expand_shape[-2], expand_shape[-1])

        # generate data for new channels
        relative_indexes = torch.arange(-self.window_extend, self.window_extend + 1, device=tensor.device)
        relative_indexes = relative_indexes.unsqueeze(-1).expand(expand_shape)
        relative_indexes_mul_state = tensor[..., 0, :, :].unsqueeze(1) * relative_indexes

        # concatenate new channels with input tensor
        tensor = torch.cat((tensor, relative_indexes, relative_indexes_mul_state), dim=1)

        return tensor

    def __getitem__(self, index: int) -> torch.Tensor:
        state_index = self.sampling_indexes[index]
        feed_forward = self.data[:, 1:, state_index - self.window_extend : state_index + self.window_extend + 1]
        if self.extend_channels:
            feed_forward = self.add_channels_to_tensor(feed_forward)
        if feed_forward.size(0) == 1:
            feed_forward = feed_forward.squeeze(0)
        return feed_forward

    def to(self, device: str) -> None:
        self.data = self.data.to(device)


class L96DataLoader(pl.LightningDataModule):
    def __init__(
        self,
        dataset: DictConfig,
        observation_model: DictConfig,
        path_to_load_data: str,
        path_to_save_data: str = None,
        train_validation_split: float = 0.75,
        shuffle_train: bool = True,
        shuffle_valid: bool = False,
        batch_size: int = 1,
        drop_last_batch: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.observation_model = hydra.utils.instantiate(observation_model)
        self.path_to_load_data = path_to_load_data
        self.path_to_save_data = path_to_save_data
        self.train_validation_split = train_validation_split
        self.shuffle_train = shuffle_train
        self.shuffle_valid = shuffle_valid
        self.batch_size = batch_size
        self.drop_last_batch = drop_last_batch
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, **kwargs) -> None:
        with h5py.File(self.path_to_load_data, "r") as file:
            simulation = torch.from_numpy(file["first_level"][:])
            training_split = int(simulation.size(1) * self.train_validation_split)

        self.train_data: Dataset = hydra.utils.instantiate(
            self.dataset,
            ground_truth_data=simulation[:, :training_split, :],
            path_to_save_data=os.path.join(self.path_to_save_data, "train") if self.path_to_save_data else None,
            observation_model=self.observation_model,
        )

        self.valid_data: Dataset = hydra.utils.instantiate(
            self.dataset,
            ground_truth_data=simulation[:, training_split:, :],
            path_to_save_data=os.path.join(self.path_to_save_data, "valid") if self.path_to_save_data else None,
            observation_model=self.observation_model,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            shuffle=self.shuffle_train,
            batch_size=self.batch_size,
            drop_last=self.drop_last_batch,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_data,
            shuffle=self.shuffle_valid,
            batch_size=self.batch_size,
            drop_last=self.drop_last_batch,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
