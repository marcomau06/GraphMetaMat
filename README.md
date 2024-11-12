# Introduction

This repo contains code both to train a lattice to curve encoder and curve to lattice policy network. All code is located in the `src` directory. The entry point for the forward model is `main_forward.py`. The entry point for the inverse model is `main_inverse.py`.

# Installation

## Requirements

Please ensure you have the below dependencies (exact versions recommended but may not be necessary):
```
bidict==0.22.0
matplotlib==3.1.2
networkx==3.1
numpy==1.24.3
pandas==1.5.1
runstats==2.0.0
scikit-learn==1.2.2
scipy==1.10.1
seaborn==0.12.2
Shapely==1.8.2
tensorboard==2.5.0
tensorboardX==2.5.1
torch==1.12.1+[your_cuda_version]
torch-geometric==2.0.4
torch-scatter==2.1.1
torch-sparse==0.6.17
tqdm==4.64.0
PyYAML==6.0
dtaidistance==2.3.10
optuna==3.3.0
```

## Models and Configurations

You can download the models and configurations from: https://drive.google.com/drive/folders/1KLJfp8dODqC7Ua4AzwF6ggcjVHnZdGvk?usp=drive_link.

Place it in the following directory structure: `GraphMetaMat/logs`

## Datasets

You can download the datasets from: https://drive.google.com/drive/folders/1Vl5Bhjhss7YOZdxGr1FWZXsZX20l2Ln5?usp=sharing.

# Usage

Update the filepath in `src/config.py` as follows:

```
SRC_DIR=/path/to/GraphMetaMat
```

Modify the filepaths in `config_dataset.yaml`:

```
dataset_RL:
    root_graph: /path/to/data_inverse or null
    root_curve: /path/to/data_inverse or null
    root_mapping: /path/to/data_inverse or null

dataset:
    root_graph: /path/to/data_forward or null
    root_curve: /path/to/data_forward or null
    root_mapping: /path/to/data_forward or null
```

Modify the filepaths in `config_general.yaml` (**We recommend creating a new log directory for each experiment**):

```
load_model: /path/to/forward_model or null
load_model_IL: /path/to/inverse_model or null
load_model_RL: /path/to/inverse_model or null
log_dir: /path/to/newly_created_log_directory
```

To choose a particular device, modify the flag in `src/config_general.yaml` as follows:

```
device: cpu             # if using CPU
device: cuda            # if using one GPU
device: cuda:[gpu_id]   # if using multiple GPUs
```

## Inference

### Stress-Strain Curve

Update the configuration in `src/config.py` as follows:
```
ETH_FULL_C_VECTOR = False
TASK = 'stress_strain'
```

Replace the config files in `src/*.yaml` with  the config files from `logs/inference/inverse_stressstrain/*/*.yaml`. Modify the file paths as described above.

### Transmission Curve

Update the configuration in `src/config.py` as follows:
```
ETH_FULL_C_VECTOR = False
TASK = 'transmission'
```

Replace the config files in `src/*.yaml` with  the config files from `logs/inference/inverse_transmission/*/*.yaml`. Modify the file paths as described above.

## Training (Experimental)

### Pretraining

Update the configuration in `src/config.py` as follows:
```
ETH_FULL_C_VECTOR = True
TASK = 'stress_strain'
```

Replace the config files in `src` with  the config files from `logs/pretraining/*.yaml`. Modify the file paths as described above.

Run the command: `$ python3 main_forward.py`

### Finetuning

Update the configuration in `src/config.py` as follows (`$task='stress_strain'` for stress strain and `$task='transmission'` for transmission):

```
ETH_FULL_C_VECTOR = False
TASK = $task
```

Replace the config files in `src` with  the config files from `logs/finetuning_stressstrain/*.yaml` for stress strain and `logs/finetuning_transmission/*.yaml` for transmission. Modify the file paths as described above.

Run the command: `$ python3 main_forward.py`

### Imitation and Reinforcement Learning


Update the configuration in `src/config.py` as follows (`$task='stress_strain'` for stress strain and `$task='transmission'` for transmission):

```
ETH_FULL_C_VECTOR = False
TASK = $task
```

Replace the config files in `src` with  the config files from `logs/ILRL_stressstrain/*.yaml` for stress strain and `logs/ILRL_transmission/*.yaml` for transmission. Modify the file paths as described above.

Run the command: `$ python3 main_inverse.py`