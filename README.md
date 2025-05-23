
# Designing metamaterials with programmable nonlinear responses and geometric constraints in graph space
![Demo Animation](Supplementary%20Video%202.gif)

This repository contains the code for GraphMetaMat — a graph-based, defect-aware framework for the inverse design of manufacturable truss metamaterials with nonlinear mechanical responses and geometric constraints, as described in [LINK_PAPER].

Below is a quick-start tutorial for running inference using a model trained on stress–strain curves.

# Quick Run

## Installation
To conduct similar analyses as in the paper, begin by cloning this repository:
```
git clone https://github.com/marcomau06/GraphMetaMat.git
```
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

Next, download the data, the trained models and configuration files, `config_general.py`, `config_dataset.py`, and `config_model.py`, from the [figshare data repository](https://doi.org/10.6084/m9.figshare.28773806). The three config files control the model, data, and directory settings. All generated outputs and trained model files are saved to a log file, specified by the `log_dir` parameter of `config_general.py`. To run inference, `log_dir` that contains the trained model should be specified.

Overwrite the `config_general.py`, `config_dataset.py`, and `config_model.py` from the `/path/to/GraphMetaMat/logs/quick_run` directory to the `/path/to/GraphMetaMat` directory. Next, set the `log_dir` in `config_general.yaml` to the directory that contains the trained model:

```
load_model: /path/to/GraphMetaMat/logs/quick_run/trained_model # NOTE: this is different for non quick_run setups...
load_model_IL: /path/to/GraphMetaMat/logs/quick_run/trained_model # NOTE: this is different for non quick_run setups...
load_model_RL: /path/to/GraphMetaMat/logs/quick_run/trained_model # NOTE: this is different for non quick_run setups...
log_dir: /path/to/GraphMetaMat/logs/quick_run
```
Next, set the path to the datasets in `config_dataset.yaml`:
```
dataset_RL:
    root_graph: /path/to/GraphMetaMat/dataset/stress_strain/standard
    root_curve: /path/to/GraphMetaMat/dataset/stress_strain/standard
    root_mapping: /path/to/GraphMetaMat/dataset/stress_strain/standard

dataset:
    root_graph: /path/to/GraphMetaMat/dataset/stress_strain/standard
    root_curve: /path/to/GraphMetaMat/dataset/stress_strain/standard
    root_mapping: /path/to/GraphMetaMat/dataset/stress_strain/standard
```

Next, set the device flag (GPU or CPU) in `/path/to/GraphMetaMat/src/config_general.yaml`:

```
device: cpu             # if using CPU
device: cuda            # if using GPU
```

To reproduce the stress-strain results on the test set (split 90/5/5, known curve space), run the model with:
```
$python3 main_inverse.py
```

You should see the following output:
```
Results:
mae: 0.0005315262824296951
mse: 2.246857320642448e-06
jaccard: 0.8847618663235556
Time taken: 2667.6254324913025s
```

If you see this output, congratulations! You have successfully ran the model and generated metamaterials with target nonlinear stress-strain curves. 

# General Usage
GraphMetaMat consists of two components: a forward model and an inverse model.
The inverse model is a policy network conditioned on a target curve. It autoregressively generates the graph representation of a truss-based metamaterial. This model is trained using a combination of imitation learning (IL) and reinforcement learning (RL).

During RL, a pre-trained forward model (structure-to-curve) predicts the mechanical response—such as a stress–strain or transmission curve—of the generated structure. The mismatch between the predicted and target curves is used to guide the policy’s optimization. Monte Carlo Tree Search (MCTS) is used at inference to improve inverse design performance.

Below are instructions to train and deploy both models.

### Forward Model
All the preset configurations in the log files in the [figshare data repository](https://doi.org/10.6084/m9.figshare.28773806) are by default for training and inference. To run training and inference, follow the same steps from [Quick Run](#quick-run) but (1) obtain the configurations from a `/path/to/GraphMetaMat/logs/*_forward` directory, (2) set `dataset` in `config_dataset.yaml` accordingly, (3) set the `load_model_IL`, `load_model_RL` and `load_model` in `config_model.yaml` following [Trained Models](#trained-models), and **(4) set `log_dir` in `config_general.yaml` to be an empty directory, where the trained model and inference results will be saved.**

Run the model with:
```
$python3 main_forward.py
```
To predict transmission curves, see [Different Types of Curves](#Transmission-Curve).

### Inverse Model

All the preset configurations in the log files in the [figshare data repository](https://doi.org/10.6084/m9.figshare.28773806) are by default for training. To run training and inference, follow the same steps from [Quick Run](#quick-run) but (1) obtain the configurations from a `/path/to/GraphMetaMat/logs/*_inverse` directory, (2) set `dataset` and `dataset_RL` in `config_dataset.yaml` accordingly, where the trained model and inference results will be saved, and (3) set the `load_model_IL`, `load_model_RL` and `load_model` in `config_model.yaml` following [Trained Models](#trained-models), and (4) **set `log_dir` in `config_general.yaml` to be an empty directory, where the trained model and inference results will be saved.**

To run only inference, follow the same steps as [Quick Run](#quick-run) but (1) obtain the configurations from a `/path/to/GraphMetaMat/logs/*_inverse` directory, (2) set `dataset` and `dataset_RL` in `config_dataset.yaml` accordingly, (3) set `load_model_IL`, `load_model_RL` and `load_model` in `config_model.yaml` following [Trained Models](#trained-models), and **(4) set `num_epochs`, `num_imitation_epochs`, and `num_iters` to be `0` in `config_general.yaml`**.

Run the model with:
```
$python3 main_inverse.py
```
To target transmission curves, see [Different Types of Curves](#Transmission-Curve).


To experiment with different setups simply change the config files: `config_general.py`, `config_dataset.py`, and `config_model.py`. Here you can adjust all hyperparameters, including network architectures, training and search settings. 

For further information, please first refer to the [paper](), the [Supplementary Information]() or reach out to [Derek Xu](mailto:derekqxu@ucla.edu) or [Marco Maurizi](mailto:marcomaurizi06@gmail.com).

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

## Benchmark models and data

To reproduce the benchmark results, please download the models and plotting scripts from the [figshare data repository](https://doi.org/10.6084/m9.figshare.28773806).
