# CODA: Combined Optimization of Dynamics and Assimilation with End-to-End Learning on Sparse Observations
A novel method for training data assimilation networks directry from sparse and noisy observations, tuning free parameters of a simulator and learning effect of unresolved processes.

## Data generation
Lorenz'96 training data can be generated using generation script from [mdml-tools](https://codebase.helmholtz.cloud/m-dml/mdml-tools/-/blob/main/mdml_tools/scripts/generate_lorenz_data.py). Our training code supports training using one long Lorenz'96 simulation that is going to be split into training and validation subsets.

Once data have been generated using the script above, you'll need to edit configuration files to set the path to the data, or override the appropriate argument from the command line.

## Running experiments
We use hydra to manage config files for our experiments. To see all availabable parameters and flags refere to corresponding experiment configs in `conf/experiment`. All parameters can be overwritten from command line.

### Data assimilation
Training data assimilation network on *Lorenz' 96 one-level* data considering forward operator known.

Example:
```bash
python main.py +experiment=data_assimilation output_dir_base_path="path/where/to/save" datamodule.path_to_load_data="path/to/load/l96/simulation.h5" l96forcing=8.0 observation_model.additional_noise_std=1.0 observation_model.random_mask_fraction=0.75 rollout_length=25 input_window_extend=25 loss_alpha=0.5 random_seed=111
```

### Parameter tuning
Training data assimilation network along estimating forcing parameter on *Lorenz' 96 one-level* data considering forward operator known.

Example:
```bash
python main.py +experiment=parameter_tuning output_dir_base_path="path/where/to/save" datamodule.path_to_load_data="path/to/load/l96/simulation.h5" observation_model.additional_noise_std=1.0 observation_model.random_mask_fraction=0.75 rollout_length=25 input_window_extend=25 loss_alpha=0.5 random_seed=111
```

### Parametrization learning
Learning an effect of a sub-grid-scale process on *Lorenz'96 two-level* data considering that sub-grid process is not represented by forward operator. This experiment is devided into two parts: pretraining data assimilation network and simultatious training of data assimilation network and parametrization.

Pretraining example:
```bash
python main.py +experiment=pretrain_data_assimilation output_dir_base_path="path/where/to/save" datamodule.path_to_load_data="path/to/load/l96/simulation.h5" l96forcing=10.0 observation_model.additional_noise_std=1.0 observation_model.random_mask_fraction=0.75 rollout_length=25 input_window_extend=25 loss_alpha=0.5 random_seed=111
```

Parametrization learning example:
```bash
python main.py +experiment=parametrization_learning output_dir_base_path="path/where/to/save" assimilation_network_checkpoint="path/to/pretrained/da/network/checkpoint" datamodule.path_to_load_data="path/to/load/l96/simulation.h5" l96forcing=10.0 observation_model.additional_noise_std=1.0 observation_model.random_mask_fraction=0.75 rollout_length=25 input_window_extend=25 loss_alpha=0.5 random_seed=111
```
