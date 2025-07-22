# Designing metamaterials with programmable nonlinear responses and geometric constraints in graph space
[![DOI](https://zenodo.org/badge/979538682.svg)](https://doi.org/10.5281/zenodo.15498444)
![Demo Animation](Supplementary%20Video%202.gif)

This repository contains the code for GraphMetaMat — a graph-based, defect-aware framework for the inverse design of manufacturable truss metamaterials with nonlinear mechanical responses and geometric constraints, as described in [paper](https://www.nature.com/articles/s42256-025-01067-x).

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

Next, download the data, checkpoints, trained models and configuration files, `config_general.py`, `config_dataset.py`, and `config_model.py`, from the [figshare data repository](https://doi.org/10.6084/m9.figshare.28773806). The three config files control the model, data, and directory settings. All generated outputs and trained model files are saved to a log file, specified by the `log_dir` parameter of `config_general.py`. To run inference, `log_dir` that contains the trained model should be specified.

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
## Running on Custom Datasets
### Forward Training + Reinforcement Learning
Step 1. Copy the configuration files from `new_dataset.zip` to `/path/to/GraphMetaMat/src`.
Step 2. Preprocess the dataset into the same format as the pickle files in [figshare data repository](https://doi.org/10.6084/m9.figshare.28773806).
```
/path/to/GraphMetaMat/dataset/our_dataset
├── train
│   ├── graphs
│   ├── curves
│   └── mapping.tsv
├── dev
│   ├── graphs
│   ├── curves
│   └── mapping.tsv
└── test
    ├── graphs
    ├── curves
    └── mapping.tsv
```
`/path/to/GraphMetaMat/dataset/our_dataset/.../graphs` is a directory containing graphs. Each graph is represented by `[GID].gpkl`. Each graph has a unique (across splits) integer `[GID]`.
Create `[GID].gpkl` with:
```
import pickle
import networkx as nx
from src.dataset_feats_node import get_node_feats
from src.dataset_feats_edge import get_edge_li, get_edge_index, get_edge_feats
g = nx.Graph() # create a networkx graph with coordinates in the range [-1,1]
edge_li = get_edge_li(g)
g.graph['node_feats'] = get_node_feats(g, edge_index)
g.graph['edge_feats'] = get_edge_feats(g, edge_li)
g.graph['edge_index'] = get_edge_index(edge_li)
g.graph['rho'] = 1.0
with open('/path/to/GraphMetaMat/dataset/our_dataset/.../graphs/[GID].gpkl', 'wb') as fp:
    pickle.dump(g, fp)
```
`/path/to/GraphMetaMat/dataset/our_dataset/.../curves` is a directory containing curves. Each curve is represented by `[CID].pkl`. Each curve has a unique (across splits) integer `[CID]`
Create `[CID].pkl` with:
```
import pickle
import numpy as np
c = np.arange(L)**2 # create a curve of shape (L,)
c_xy = np.stack((np.arange(c.shape[0]), curve), axis=-1)
c_obj = \
    {
        'curve': c_xy,
        'cid': CID,
        'is_monotonic': True
    }
with open('/path/to/GraphMetaMat/dataset/our_dataset/.../curves/[CID].gpkl', 'wb') as fp:
    pickle.dump(c_obj, fp)
```
`/path/to/GraphMetaMat/dataset/our_dataset/.../mapping.tsv` is a list of tab seperated `[GID]` `[CID]` pairs:
```
[GID0]\t[CID0]
[GID1]\t[CID1]
[GID2]\t[CID2]
...
```
Step 3.1. Set up the data directories in `/path/to/GraphMetaMat/src/config_dataset.yaml`:
```
dataset_RL:
    root_graph: /path/to/GraphMetaMat/dataset/our_dataset
    root_curve: /path/to/GraphMetaMat/dataset/our_dataset
    root_mapping: /path/to/GraphMetaMat/dataset/our_dataset
dataset:
    root_graph: /path/to/GraphMetaMat/dataset/our_dataset
    root_curve: /path/to/GraphMetaMat/dataset/our_dataset
    root_mapping: /path/to/GraphMetaMat/dataset/our_dataset
```
Step 3.2. Set up the output directory in `/path/to/GraphMetaMat/src/config_general.yaml` (create an empty directory, `forward_model`, where the model output will be saved):
```
log_dir: /path/to/GraphMetaMat/logs/forward_model
```
Step 3.3. Next, set the device flag (GPU or CPU) in `/path/to/GraphMetaMat/src/config_general.yaml`:
```
device: cpu             # if using CPU
device: cuda            # if using GPU
```
Step 4.1. Train the forward model with:
```
$python3 main_forward.py
```
Step 4.2. Set up the checkpoint directories in `/path/to/GraphMetaMat/src/config_general.yaml`:
```
load_model: /path/to/GraphMetaMat/logs/forward_model/model_epoch_snapshot_4.pt
load_model_IL: null
load_model_RL: null
```
Step 4.3. Set up the output directory in `/path/to/GraphMetaMat/src/config_general.yaml` (create an empty directory, `inverse_model`, where the model output will be saved):
```
log_dir: /path/to/GraphMetaMat/logs/inverse_model
```
Step 5 Train the inverse model by following [Inverse Model](#asdf), **using the configuration setup from Step 2**.
```
$python3 main_inverse.py
```
### Include Pretrained Model
After Step 3.3, set up the checkpoint directories in `/path/to/GraphMetaMat/src/config_general.yaml`:
```
load_model: /path/to/GraphMetaMat/checkpoints/pretrain.pt
load_model_IL: null
load_model_RL: null
```
### Add Imitation Learning
Imitation learning reads from a seperate data file `[GID]_polyhedron.gpkl` corresponding to `[GID].gpkl` in `/path/to/GraphMetaMat/dataset/our_dataset/.../graphs`.
Create `[GID]_polyhedron.gpkl` with:
```
with open(os.path.join(pn_graphs, f'{gid}.gpkl'), 'rb') as fp:
    g = pickle.load(fp)
untesselate(g, start='unit_cell', end='tetrahedron', rm_redundant=True)
assert g_tetrahedron.number_of_nodes() <= 4, \
    f'tetrahedron has {g_tetrahedron.number_of_nodes()} nodes; graph has {g.number_of_nodes()} nodes'
with open(os.path.join(pn_graphs, f'{gid}_polyhedron.gpkl'), 'wb') as fp:
    pickle.dump(g_tetrahedron, fp)
```
After Step 4.3, add imitation learning in `/path/to/GraphMetaMat/src/config_general.yaml`:
```
inverse:
    train_config:
        num_imitation_epochs: 128
...
```

## Benchmark models and data

To reproduce the benchmark results, please download the models and plotting scripts from the [figshare data repository](https://doi.org/10.6084/m9.figshare.28773806).

## Citation
If this code is useful for your research, please cite our [paper](https://www.nature.com/articles/s42256-025-01067-x).
