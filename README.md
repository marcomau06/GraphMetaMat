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

## Datasets

You can download the datasets from: https://drive.google.com/drive/folders/1Vl5Bhjhss7YOZdxGr1FWZXsZX20l2Ln5?usp=sharing.

### Preprocessing ETH Dataset (Pretraining Dataset)

Save the dataset to a directory in the following style: 

```
/[path_to_data]/ETH/Fine_tuning_data/ETHunifying_cubic_100k_rho15.json
```

Open `dataset_preprocessing.py` and ensure the following variables are set:

```
PSU_dataset = False
ETH_dataset = True
dn_dataset = '/[path_to_data]/ETH'
mix_dataset = True
```

Run `$ python3 dataset_preprocessing.py`.

### Preprocessing Stress-Strain Curve Dataset (Fine-tuning Dataset and RL Dataset)

Save the dataset to a directory in the following style: 

```
/[path_to_data]/StressStrain/Fine_tuning_data/c_N3_4_D5_FLX_str30_2kGs_randrho.json
/[path_to_data]/StressStrain/Fine_tuning_data/cubic_N3_4_D5_FLEXIBLE_strain30pct_rho5_15_25pct_400graphs.json
```

Open `dataset_preprocessing.py` and ensure the following variables are set:

```
PSU_dataset = False
ETH_dataset = False
dn_dataset = '/[path_to_data]/StressStrain'
mix_dataset = False
```

Run `$ python3 dataset_preprocessing.py`.

<!---

### Preprocessing Transmission Curve Dataset (Fine-tuning Dataset)

Save the dataset to a directory in the following style: 

```
/[path_to_data]/Transmission/Fine_tuning_data/cubic_N3_4_D5_TMPTA_TL_random_rho_1333cells.json
```

Open `dataset_preprocessing.py` and ensure the following variables are set:

```
PSU_dataset = True
ETH_dataset = False
dn_dataset = '/[path_to_data]/Transmission'
mix_dataset = False
```

Run `$ python3 dataset_preprocessing.py`.

-->

## Specifying CPU or GPU

Set the flag in `src/config_general.yaml` as follows:

```
device: cpu             # if using CPU
device: cuda            # if using one GPU
device: cuda:[gpu_id]   # if using multiple GPUs
```

# Usage

## Forward Model Instructions

Modify the config files based on which dataset is used for training.

### Pretraining

Open the specified `src/*.yaml` files and ensure the following variables are set:

```
# config_general.yaml
load_model: null
log_dir: /[existing_empty_log_directory]

# config_dataset.yaml
dataset:
    curve_norm_cfg:
        curve_method: max
    root_graph: /[path_to_data]/ETH/preprocessed_unitcell_True
    root_curve: /[path_to_data]/ETH/preprocessed_unitcell_True
    root_mapping: /[path_to_data]/ETH/preprocessed_unitcell_True

# config_model.yaml
forward_model:
  loss_coeff:
    magnitude_coeff: 1.0
    shape_coeff: 0.0
```

Run `$ python3 main_forward.py`.

### Stress-Strain Fine-tuning

Open the specified `src/*.yaml` files and ensure the following variables are set:

```
# config_general.yaml
load_model: /[pretrained_models_log_directory] # to turn off pretraining set to null
log_dir: /[existing_empty_log_directory]

# config_dataset.yaml
dataset:
    curve_norm_cfg:
        curve_method: max
    root_graph: /[path_to_data]/ETH/preprocessed_unitcell_True
    root_curve: /[path_to_data]/ETH/preprocessed_unitcell_True
    root_mapping: /[path_to_data]/ETH/preprocessed_unitcell_True

# config_model.yaml
forward_model:
  loss_coeff:
    magnitude_coeff: 0.0 # tunable but should be >0.0
    shape_coeff: 1.0 # tunable but should be >0.0
```

Run `$ python3 main_forward.py`.

<!---

### Transmission Fine-tuning

Open the specified `src/*.yaml` files and ensure the following variables are set:

```
# config_general.yaml
load_model: /[pretrained_models_log_directory] # to turn off pretraining set to null
log_dir: /[existing_empty_log_directory]

# config_dataset.yaml
dataset:
    curve_norm_cfg:
        curve_method: max
    root_graph: /[path_to_data]/ETH/preprocessed_unitcell_True
    root_curve: /[path_to_data]/ETH/preprocessed_unitcell_True
    root_mapping: /[path_to_data]/ETH/preprocessed_unitcell_True

# config_model.yaml
forward_model:
  loss_coeff:
    magnitude_coeff: 0.0
    shape_coeff: 1.0
```

Run `$ python3 main_forward.py`.

-->


## Inverse Model Instructions

Modify the config files based on which dataset is used for training.

### Stress-Strain RL Training

Open the specified `src/*.yaml` files and ensure the following variables are set:

```
# config_general.yaml
load_model: /[finetuned_models_log_directory]
log_dir: /[existing_empty_log_directory]
inverse:
  search:
    magnitude_coeff: -1.0 # tunable but should be <0.0
    shape_coeff: -1.0 # tunable but should be <0.0

# config_dataset.yaml
dataset:
    curve_norm_cfg:
        curve_method: max
    root_graph: /[path_to_data]/ETH/preprocessed_unitcell_True
    root_curve: /[path_to_data]/ETH/preprocessed_unitcell_True
    root_mapping: /[path_to_data]/ETH/preprocessed_unitcell_True
```

Run `$ python3 main_inverse.py`.

<!---

### Transmission RL Training

Open the specified `src/*.yaml` files and ensure the following variables are set:

```
# config_general.yaml
load_model: /[finetuned_models_log_directory]
log_dir: /[existing_empty_log_directory]
inverse:
  search:
    magnitude_coeff: 0.0
    shape_coeff: -1.0

# config_dataset.yaml
dataset:
    curve_norm_cfg:
        curve_method: max
    root_graph: /[path_to_data]/ETH/preprocessed_unitcell_True
    root_curve: /[path_to_data]/ETH/preprocessed_unitcell_True
    root_mapping: /[path_to_data]/ETH/preprocessed_unitcell_True
```

Run `$ python3 main_inverse.py`.

-->
