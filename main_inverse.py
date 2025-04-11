import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from copy import deepcopy

from src.dataset import DataLoaderFactory
from src.generative_curve.model import ModelEnsemble, Model
from src.generative_graph.model_v2 import PolicyNetwork
from src.generative_graph.train import train, pretrain_imitation, TrainConfig, SearchConfig
from src.generative_graph.test import test, plot_pred
from src.generative_graph.env_v2 import get_reward_helper, get_curve_helper, get_jaccard
from src.config import args
from src.utils import CObj, log_dir, to_list, get_optimizer
from pprint import pformat
from  collections import defaultdict

import numpy as np
import os
import torch
import time
from torch.optim.lr_scheduler import LinearLR
from src.generative_curve.pretrain import write_model

DEVICE = args['device']
KERNEL_CONFIG = 'inverse'

def fit_train_data_kernel(dataset_test, dataset_kernel):  # , g_dict_li, c_dict_li, c_dict_bg, topk=5):
    print('running fit train data kernel')
    kernel_curves = torch.tensor(np.stack([c.c[:, 1] for (_, c) in dataset_kernel.dataset], axis=0))
    test_curves = torch.tensor(np.stack([c.c[:, 1] for (_, c) in dataset_test.dataset], axis=0))

    g_train_best_li, curve_train_best_li = [], []
    best_indices = []
    for test_curve in tqdm(test_curves):
        idx = int(torch.argmax(get_jaccard(test_curve.view(1,-1,1), kernel_curves.view(*kernel_curves.shape, 1))))
        g, c = dataset_kernel.dataset[idx]
        best_indices.append(best_indices)
        g_train_best_li.append(g)
        curve_train_best_li.append(c)

    return g_train_best_li, curve_train_best_li, best_indices

def apply_train_kernel(g_train_best_li, curve_train_best_li, g_dict_li, c_dict_li):
    assert len(g_train_best_li) == len(curve_train_best_li) == len(g_dict_li) == len(c_dict_li)
    for k, (g_kernel, c_kernel) in enumerate(zip(g_train_best_li, curve_train_best_li)):
        g_dict_li[k][f'TrainGraph'] = g_kernel.g
        c_dict_li[k][f'TrainCurve'] = (c_kernel, None, None)

def digitize_curve_np(digitize_cfg, curve):
    '''
    Args:
        curve: 2xL curve
    Returns:
        curve_digitized: 2xn_freq curve
    '''
    bins = np.array([float(x) for x in digitize_cfg['bins']])
    n_freq = digitize_cfg['n_freq']
    blocksize = int(curve.shape[1] / n_freq)
    assert curve.shape[1] % n_freq == 0
    out = np.median(np.digitize(curve, bins).reshape(curve.shape[0], -1, blocksize), axis=-1).astype(int)
    return out

def plot_curves():
    plt.clf()
    root_pn = log_dir
    result_pn = f'{root_pn}/results.pkl'
    with open(result_pn, 'rb') as fp:
        c = pickle.load(fp)
    if 'digitize_cfg' in args['dataset'] and args['dataset']['digitize_cfg'] is not None:
        c_true = torch.stack([
            torch.tensor(
                digitize_curve_np(args['dataset']['digitize_cfg'], np.expand_dims(x[0], axis=0))[0]
            ) for x in c['curve_true_li']], dim=0)
        c_pred = torch.stack([
            torch.tensor(
                digitize_curve_np(args['dataset']['digitize_cfg'], np.expand_dims(x[0], axis=0))[0]
            ) for x in c['curve_RL_pred_li']], dim=0)
    else:
        c_true = torch.stack([torch.tensor(x[0]) for x in c['curve_true_li']], dim=0)
        c_pred = torch.stack([
            (torch.tensor(x[0]) if type(x[0]) == np.ndarray else x[0]).view(-1)
            for x in c['curve_RL_pred_li']], dim=0)

    if 'digitize_cfg' in args['dataset'] and args['dataset']['digitize_cfg'] is not None:
        diff_pred = (c_pred == c_true).float().mean(-1)
    else:
        diff_pred = (torch.abs(c_pred - c_true) / (
                    torch.max(c_true, dim=-1)[0] - torch.min(c_true, dim=-1)[0]).unsqueeze(-1)).mean(
            dim=-1)

    plt.hist(diff_pred, label=f'pred, NMAE={diff_pred.mean()}', alpha=0.5)
    plt.legend()
    plt.savefig(f'{root_pn}/tmp.png')
    plt.clf()
    plt.savefig(f'{root_pn}/tmp_max.png')
    plt.clf()
    plt.savefig(f'{root_pn}/tmp_min.png')
    plt.clf()

def main():
    ######## Seed for reproducible results ##########
    import random
    import numpy as np
    rs=1
    random.seed(rs)
    np.random.seed(rs)
    torch.manual_seed(rs)
    torch.cuda.manual_seed(rs)
    #################################################

    ############### Datasets loading ###################
    tmp = args['dataset']['train_split']
    tmp2 = args['dataset']['augment_curve']
    tmp3 = args['dataset']['augment_graph']
    args['dataset']['train_split'] = 'train'
    args['dataset']['augment_curve'] = False
    args['dataset']['augment_graph'] = False
    dlf_kernel = DataLoaderFactory(**args['dataset'])
    dataset_forward = dlf_kernel.get_train_dataset()
    dataset_forward.apply_patch = False
    args['dataset']['train_split'] = tmp
    args['dataset']['augment_curve'] = tmp2
    args['dataset']['augment_graph'] = tmp3

    args['dataset'].update(args['dataset_RL'])
    dlf = DataLoaderFactory(**args['dataset'])
    dlf.train_dataset.g_stats = dataset_forward.dataset.g_stats
    dlf.train_dataset.c_stats = dataset_forward.dataset.c_stats
    dataset_train = dlf.get_train_dataset()
    dataset_valid = dlf.get_valid_dataset()
    dataset_test = dlf.get_test_dataset()
    #####################################################

    ########### Creating configuration ###################
    t0 = time.time()
    search_cfg = SearchConfig(**args['inverse']['search'])
    train_cfg = TrainConfig(**args['inverse']['train_config'])
    #####################################################

    ############# Load forward model ####################
    model_forward = Model.init_from_cfg(dataset=dataset_train, **args['forward_model'])

    assert 'load_model' in args and args['load_model'] != 'None' and args['load_model'] is not None

    device = 'cuda' if DEVICE == 'cuda' else 'cpu'
    if args['forward']['train_config']['use_snapshot'] is not None:
        pn_load = os.path.split(args['load_model'])[0]
        model_forward = ModelEnsemble.from_path(pn_load, model_forward)
    else:
        model_forward.load_state_dict(torch.load(args['load_model'], map_location=device))
    model_forward = model_forward.to(torch.device(device))
    #####################################################

    ############# Load inverse model ####################
    policy = PolicyNetwork.init_from_cfg(dataset_train, **args['policy_network'], search_cfg=search_cfg)
    policy.train()
    policy = policy.to(torch.device(device))
    optimizer = get_optimizer(model=policy, **args['optimizer'])
    if train_cfg.use_scheduler is not None:
        total_iters = int(train_cfg.num_iters / train_cfg.num_iters_per_train)
        optimizer = LinearLR(optimizer, **train_cfg.use_scheduler, total_iters=total_iters)
    #####################################################

    ############# Training inverse model ################
    reward_train_log = None
    debugging_log = None
    if 'load_model_RL' in args and args['load_model_RL'] is not None:
        policy.load_state_dict(torch.load(args['load_model_RL'], map_location=device))
    else:
        ################# Imitation learning ###############
        if 'load_model_IL' in args and args['load_model_IL'] is not None:
            policy.load_state_dict(torch.load(args['load_model_IL'], map_location=device))
        else:
            policy, loss_log = \
                pretrain_imitation(
                    dataset_train=dataset_train,
                    dataset_forward=dataset_forward,
                    policy=policy,
                    optimizer=optimizer,
                    train_cfg=train_cfg,
                    search_cfg=search_cfg,
                    device=device)
            write_model(policy, log_dir, 'imitation')
            plt.plot(torch.arange(len(loss_log)), loss_log)
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.savefig(os.path.join(log_dir, 'IL_loss.png'))

        ################# Reinforcement learning ###############
        if train_cfg.num_iters > 0:
            optimizer = get_optimizer(model=policy, **args['optimizer_RL'])
            if 'optimizer_decay' in args and args['optimizer_decay'] is not None:
                optimizer = \
                    torch.optim.lr_scheduler.ExponentialLR(
                        optimizer=optimizer, gamma=args['optimizer_decay'])
            # dataset_train.dataset.augment_curve = 'true'
            policy, reward_train_log, reward_valid_log, debugging_log = \
                train(
                    dataset_train=dataset_train,
                    dataset_valid=dataset_valid,
                    dataset_forward=dataset_forward,
                    policy=policy,
                    model_surrogate=model_forward,
                    optimizer=optimizer,
                    train_cfg=train_cfg, search_cfg=search_cfg,
                    device=DEVICE, node_feat_cfg=args['dataset']['node_feat_cfg'],
                    edge_feat_cfg=args['dataset']['edge_feat_cfg']
                )
    #####################################################

    ###################### Plotting #####################
    num_iters_per_valid = args['inverse']['train_config']['num_iters_per_valid']
    num_iters_per_train = args['inverse']['train_config']['num_iters_per_train']
    print('plotting rewards')
    if reward_train_log is not None:
        plt.clf()
        plt.plot(torch.arange(len(reward_train_log)), reward_train_log, label='train')
        plt.plot(num_iters_per_valid * torch.arange(len(reward_valid_log)), reward_valid_log, label='valid')
        plt.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Reward')
        plt.savefig(os.path.join(log_dir, 'reward.png'))
    if debugging_log is not None:
        for k, v in debugging_log.items():
            plt.clf()
            plt.plot(num_iters_per_train * torch.arange(len(v)), v)
            plt.xlabel('Iterations')
            plt.ylabel(k)
            plt.savefig(os.path.join(log_dir, f'{k}.png'))
    #####################################################

    ############ Find the best train match #############
    if KERNEL_CONFIG == 'forward':
        g_train_best_li, curve_train_best_li, best_indices_li = \
            fit_train_data_kernel(dataset_test, dataset_forward)
    elif KERNEL_CONFIG == 'inverse':
        g_train_best_li, curve_train_best_li, best_indices_li = \
            fit_train_data_kernel(dataset_test, dataset_train)
    else:
        assert KERNEL_CONFIG is None
        g_train_best_li, curve_train_best_li, best_indices_li = None, None, None
    #####################################################

    ############## Testing inverse model ################
    reward, reward_min, reward_max, plot_obj = \
        test(
            dataset_test, dataset_forward, policy, model_forward, search_cfg=search_cfg,
            device=DEVICE, skip_g_prog=False, num_runs=search_cfg.num_runs_test, g_train_best_li=g_train_best_li
        )

    if KERNEL_CONFIG is not None:
        assert g_train_best_li is not None and curve_train_best_li is not None
        graph_progress_li, graph_pred_li, g_dict_li, c_dict_li, c_dict_bg = plot_obj
        apply_train_kernel(g_train_best_li, curve_train_best_li, g_dict_li, c_dict_li)
        plot_obj = graph_progress_li, graph_pred_li, g_dict_li, c_dict_li, c_dict_bg
    #####################################################

    ################ Exporting results ##################
    if 'export_results' in args and args['export_results']:
        graph_progress_li, graph_pred_li, g_dict_li, c_dict_li, *_ = plot_obj
        graph_true_li = [g_dict['True'] for g_dict in g_dict_li]
        curve_pred_li = [c_dict['FwdModel(Pred Graph)'] for c_dict in c_dict_li]
        curve_tred_li = [c_dict['FwdModel(True Graph)'] for c_dict in c_dict_li]
        curve_true_li = [c_dict['True Curve'] for c_dict in c_dict_li]
        if KERNEL_CONFIG is not None:
            assert g_train_best_li is not None and curve_train_best_li is not None
            graph_kernel_li = [g_dict['TrainGraph'] for g_dict in g_dict_li]
            curve_kernel_li = [c_dict['TrainCurve'] for c_dict in c_dict_li]
        else:
            curve_kernel_li, graph_kernel_li = None, None
        export_obj = \
            {
                'graph_progress_li': graph_progress_li,
                'graph_pred_li': graph_pred_li,
                'graph_true_li': graph_true_li,
                'graph_kernel_li':graph_kernel_li,
                'curve_RL_pred_li': curve_pred_li,
                'curve_pred_li': curve_tred_li,
                'curve_true_li': curve_true_li,
                'curve_kernel_li':curve_kernel_li
            }
        with open(os.path.join(log_dir, f'results.pkl'), "wb") as f:
            pickle.dump(export_obj, f)
    #####################################################

    print(reward_train_log)

    print('Results:')
    metric_dict = defaultdict(list)
    for curve_dict in plot_obj[3]:
        from src.generative_graph.env_v2 import get_jaccard
        metric_dict['mae'].append(np.mean(np.abs(curve_dict['FwdModel(Pred Graph)'][0] - curve_dict['True Curve'][0])))
        metric_dict['mse'].append(np.mean((curve_dict['FwdModel(Pred Graph)'][0] - curve_dict['True Curve'][0])**2))
        metric_dict['jaccard'].append(get_jaccard(
            torch.tensor(curve_dict['FwdModel(Pred Graph)'][0]).view(1,-1,1),
            torch.tensor(curve_dict['True Curve'][0]).view(1,-1,1)).item())
    for k, v in metric_dict.items():
        print(f'{k}: {np.mean(np.array(v))}')


    print(f'Time taken: {time.time() - t0}s')

if __name__ == '__main__':
    main()
    plot_curves()