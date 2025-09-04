# stochastic_CODA

This project studies stochastic extensions of the original (deterministic) CODA model:

Zinchenko, Vadim, and David S. Greenberg. "Combined Optimization of Dynamics and Assimilation with End-to-End Learning on Sparse Observations." arXiv preprint arXiv:2409.07137 (2024).

The code is based on the one from this original work, which is available here: https://codebase.helmholtz.cloud/m-dml/hidden-process-learning

**Usage:** 

One can generate a Lorenz-96 dataset using the [mdml-tools](https://codebase.helmholtz.cloud/m-dml/mdml-tools/-/blob/main/mdml_tools/scripts/generate_lorenz_data.py) repository, or directly download pre-generated data from [here](https://drive.google.com/drive/folders/10f9RGCrtRD97OaQTyPr-jMm2k-o6i5PA?usp=sharing).

Afterwards, stochastic models can be trained by running the "main.py" script with the appropriate arguments.

For example, one can train a single model with a dropout of probability p=0.2 with:
```bash
python main.py +experiment=data_assimilation output_dir_base_path="." datamodule.path_to_load_data="data/L96_small.h5" rollout_length=25 input_window_extend=25 loss_alpha=0.5 random_seed=111 assimilation_network.dropout=0.2
```

A model parameterizing a diagonal Gaussian distribution conditioned on a window of observations can be trained with:
```bash
python main.py +experiment=data_assimilation_gaussian output_dir_base_path="." datamodule.path_to_load_data="data/L96_small.h5" rollout_length=20 input_window_extend=25 loss_alpha=0.4 random_seed=111
```
One can also do the same for the parameterization of a Gaussian distribution with a diagonal plus low-rank covariance matrix, with:
```bash
python main.py +experiment=data_assimilation_gaussian_LR output_dir_base_path="." datamodule.path_to_load_data="data/L96_small.h5" rollout_length=20 input_window_extend=25 loss_alpha=0.4 random_seed=111
```
