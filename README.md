This repo contains code both to train a lattice to curve encoder and curve to lattice policy network. All code is located in the `src` directory. The entry point for the forward model is `main_forward.py`. The entry point for the inverse model is `main_inverse.py`.

# Quick Run

## Installation

### Requirements

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

### Models and Configurations

Please download the models and configurations from: https://drive.google.com/drive/folders/1YfSF-PbeHZ1IElxiN_wO-pMEfKGIWvZU

Place it in the following directory structure: 
```
/path/to/GraphMetaMat/logs
```

### Weights

Please download the model checkpoints from: https://drive.google.com/drive/folders/1VmB7y9wTI0vSrpx0g5lh4TVQgIC6UZr8


### Datasets

Please download the preprocessed datasets (e.g., `stress_strain`) datasets from: https://drive.google.com/drive/folders/13ga9DFFtsHcHGPEgc3eeJBqvX4KlEJK-

Place it in the following directory structure: 

```
/path/to/GraphMetaMat/dataset/stress_strain
```

## Quick Run

We specify all model and data configurations for metamaterial generation in 3 configuration files: `config_general.py`, `config_dataset.py`, and `config_model.py`. For the quick-run tutorial, we will run inference with a trained model on the stress-strain dataset. All generated otuputs and trained model files are saved to a log file. This is specified in the `log_dir` parameter of `config_general.py`. To run just inference, we need to specify `log_dir` that contains the trained model. We provide the trained models in the log files in [Models and Configurations](#models-and-configurations).

First, overwrite the `config_general.py`, `config_dataset.py`, and `config_model.py` from the `/path/to/GraphMetaMat/logs/quick_run` directory to the `/path/to/GraphMetaMat` directory.

Next, set the `log_dir` in `config_general.yaml` to the directory that contains the trained model:

```
load_model: /path/to/GraphMetaMat/logs/quick_run/trained_model # NOTE: this is different for non quick_run setups...
load_model_IL: /path/to/GraphMetaMat/logs/quick_run/trained_model # NOTE: this is different for non quick_run setups...
load_model_RL: /path/to/GraphMetaMat/logs/quick_run/trained_model # NOTE: this is different for non quick_run setups...
log_dir: /path/to/GraphMetaMat/logs/quick_run
```

Next, set the path to the datasets in `config_dataset.yaml`:

```
dataset_RL:
    root_graph: /path/to/GraphMetaMat/dataset/stress_strain/data_inverse
    root_curve: /path/to/GraphMetaMat/dataset/stress_strain/data_inverse
    root_mapping: /path/to/GraphMetaMat/dataset/stress_strain/data_inverse

dataset:
    root_graph: /path/to/GraphMetaMat/dataset/stress_strain/data_forward
    root_curve: /path/to/GraphMetaMat/dataset/stress_strain/data_forward
    root_mapping: /path/to/GraphMetaMat/dataset/stress_strain/data_forward
```

Next, set the device flag (GPU or CPU) in `/path/to/GraphMetaMat/src/config_general.yaml`:

```
device: cpu             # if using CPU
device: cuda            # if using GPU
```

Run the model with:
```
$python3 main_inverse.py
```

You should see the following output, which reproduces our results from the stress-strain experiments:
```
TODO
```

If you see this output, congratulations! You have successfully ran the model.

# General Usage

Please first follow the steps in [Quick Run](#quick-run) to set up environment, download models, download data, and run inference.

For transmission datasets, see [Transmission Curves](#models-and-configurations).

## Load Trained Model and Run Inference
### Forward Model

See [Run Training and Inference](#run-training-and-inference).

### Inverse Model

To run inference, follow the same steps as [Quick Run](#quick-run) but (1) obtain the configurations from a `/path/to/GraphMetaMat/logs/*_inverse` directory, (2) set `dataset` and `dataset_RL` in `config_dataset.yaml` accordingly, (3) set `load_model_IL`, `load_model_RL` and `load_model` in `config_model.yaml` following [Trained Models](#trained-models), and **(4) set `num_epochs`, `num_imitation_epochs`, and `num_iters` to be `0` in `config_general.yaml`**.

## Run Training and Inference [Experimental]
### Forward Model

All the preset configurations in the log files from [Models and Configurations](#models-and-configurations) are by default for training and inference. 

To run training and inference, follow the same steps from [Quick Run](#quick-run) but (1) obtain the configurations from a `/path/to/GraphMetaMat/logs/*_forward` directory, (2) set `dataset` in `config_dataset.yaml` accordingly, (3) set the `load_model_IL`, `load_model_RL` and `load_model` in `config_model.yaml` following [Trained Models](#trained-models), and **(4) set `log_dir` in `config_general.yaml` to be an empty directory, where the trained model and inference results will be saved.**

Run the model with:
```
$python3 main_forward.py
```

### Inverse Model

All the preset configurations in the log files from [Models and Configurations](#models-and-configurations) are by default for training. 

To run training and inference, follow the same steps from [Quick Run](#quick-run) but (1) obtain the configurations from a `/path/to/GraphMetaMat/logs/*_inverse` directory, (2) set `dataset` and `dataset_RL` in `config_dataset.yaml` accordingly, where the trained model and inference results will be saved, and (3) set the `load_model_IL`, `load_model_RL` and `load_model` in `config_model.yaml` following [Trained Models](#trained-models), and (4) **set `log_dir` in `config_general.yaml` to be an empty directory, where the trained model and inference results will be saved.**

Run the model with:
```
$python3 main_inverse.py
```

## Trained Models

We store all the saved models under the following directory: `logs/trained_models`. Please set the `load_model_IL`, `load_model_RL` and  directories in `config_model.yaml` accordingly.

### Forward Model

`load_model` FLAG (**Training and Inference**): Pretrained forward models will have the suffix `*_pt`. Setting this FLAG will start training from the pretrained checkpoint.

`load_model` FLAG (**Inference-Only**): Finetuned ensemble of forward models will have the suffices `*_ensemble`. Setting this FLAG will bypass training by loading the trained ensemble.

### Inverse Model

`load_model` FLAG: Finetuned ensemble of forward models will have the suffices `*_ensemble`. This will be used as the surrogate model during RL training and inference.

`load_model_IL` FLAG (**Optional**): IL inverse models will have the suffix `*_IL`. Setting this FLAG will bypass just IL inverse model training by loading a trained IL inverse model.

`load_model_RL` FLAG (**Inference-Only**): RL inverse models will have the suffix `*_RL`. Setting this FLAG will bypass both IL and RL inverse model training by loading a trained RL inverse model.

_For full training and inference, set both `load_model_IL` and `load_model_RL` to `null`._

## Different Types of Curves

### Stress-Strain Curve

Update the configuration in `src/config.py` as follows:
```
ETH_FULL_C_VECTOR = False
TASK = 'stress_strain'
```

### Transmission Curve

Update the configuration in `src/config.py` as follows:
```
ETH_FULL_C_VECTOR = False
TASK = 'transmission'
```

### Pretraining [Deprecated]

Update the configuration in `src/config.py` as follows:
```
ETH_FULL_C_VECTOR = True
TASK = 'stress_strain'
```
# Benchmark models and data

To reproduce the benchmark results, please download the models and plotting scripts from https://drive.google.com/drive/folders/1takiWy7GFp5SNolhbSbNfM4siOQqOEXP, and the corresponding datasets from https://drive.google.com/drive/folders/1FlWRNBnQCLLWkXHp7CXuCT-A5HZPrxSk .
To reproduce the benchmark results, please download models, plotting scripts, and datasets from:  and 
