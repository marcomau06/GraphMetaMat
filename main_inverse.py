import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from copy import deepcopy

from src.dataset import DataLoaderFactory
from src.generative_curve.model import ModelEnsemble, Model
from src.generative_graph.model_v2 import PolicyNetwork
from src.generative_graph.train import train, pretrain_imitation, TrainConfig, SearchConfig
from src.generative_graph.test import test, plot_pred
from src.generative_graph.env_v2 import get_reward_helper, get_curve_helper
from src.config import args
from src.utils import CObj, log_dir, to_list, get_optimizer
from pprint import pformat

import numpy as np
import os
import torch
import time
from torch.optim.lr_scheduler import LinearLR
from src.generative_curve.pretrain import write_model

DEVICE = args['device']

def main():
    ######## Seed for reproducible results ##########
    import random
    import numpy as np
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    #################################################
    
    t0 = time.time()
    # save_cfg()

    ############# Datasets loading ####################
    tmp = args['dataset']['train_split']
    tmp2 = args['dataset']['augment_curve']
    tmp3 = args['dataset']['augment_graph']
    args['dataset']['train_split'] = 'train'
    args['dataset']['augment_curve'] = False
    args['dataset']['augment_graph'] = False
    dlf_kernel = DataLoaderFactory(**args['dataset'], apply_patch=True)
    dataset_kernel = dlf_kernel.get_train_dataset()
    dataset_kernel.apply_patch = True
    args['dataset']['train_split'] = tmp
    args['dataset']['augment_curve'] = tmp2
    args['dataset']['augment_graph'] = tmp3
    # dataset_valid = dlf.get_valid_dataset()
    # dataset_test = dlf.get_test_dataset()

    args['dataset'].update(args['dataset_RL'])
    dlf = DataLoaderFactory(**args['dataset'])
    dlf.train_dataset.g_stats = dataset_kernel.dataset.g_stats
    dlf.train_dataset.c_stats = dataset_kernel.dataset.c_stats
    dataset_train = dlf.get_train_dataset()
    # dataset_train = merge_dataset(dataset_train, dataset_kernel)
    dataset_valid = dlf.get_valid_dataset()
    dataset_test = dlf.get_test_dataset()
    ###################################################

    search_cfg = SearchConfig(**args['inverse']['search'])
    train_cfg = TrainConfig(**args['inverse']['train_config'])

    ############# Forward model loading ####################
    model_forward = Model.init_from_cfg(dataset=dataset_train, **args['forward_model'])

    assert 'load_model' in args and args['load_model'] != 'None' and args['load_model'] is not None

    device = 'cuda' if DEVICE == 'cuda' else 'cpu'
    if args['forward']['train_config']['use_snapshot'] is not None:
        pn_load = os.path.split(args['load_model'])[0]
        model_forward = ModelEnsemble.from_path(pn_load, model_forward)
    else:
        model_forward.load_state_dict(torch.load(args['load_model'], map_location=device))
    model_forward = model_forward.to(torch.device(device))
    ###################################################

    # Training
    policy = PolicyNetwork.init_from_cfg(dataset_train, **args['policy_network'], search_cfg=search_cfg)
    policy.train()
    policy = policy.to(torch.device(device))
    optimizer = get_optimizer(model=policy, **args['optimizer'])
    if train_cfg.use_scheduler is not None:
        total_iters = int(train_cfg.num_iters / train_cfg.num_iters_per_train)
        optimizer = LinearLR(optimizer, **train_cfg.use_scheduler, total_iters=total_iters)

    reward_train_log = None
    debugging_log = None
    if 'load_model_RL' in args and args['load_model_RL'] is not None:
        policy.load_state_dict(torch.load(args['load_model_RL'], map_location=device))
    else:
        if 'load_model_IL' in args and args['load_model_IL'] is not None:
            policy.load_state_dict(torch.load(args['load_model_IL'], map_location=device))
        else:
            #################### Imitation learning #################
            policy, loss_log = \
                pretrain_imitation(dataset_train, policy, optimizer, train_cfg, search_cfg, device)
            write_model(policy, log_dir, 'imitation')
            plt.plot(torch.arange(len(loss_log)), loss_log)
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.savefig(os.path.join(log_dir, 'IL_loss.png'))

        ################### Add UNLABELLED curves ###################
        # if 'dataset_RL' in args and args['dataset_RL'] is not None:
        #     dataset_train = augment_dataset_RL(dataset_train)

        #################### RL #################
        if train_cfg.num_iters > 0:
            optimizer = get_optimizer(model=policy, **args['optimizer_RL'])
            if 'optimizer_decay' in args and args['optimizer_decay'] is not None:
                optimizer = \
                    torch.optim.lr_scheduler.ExponentialLR(
                        optimizer=optimizer, gamma=args['optimizer_decay'])
            # dataset_train.dataset.augment_curve = 'true'
            policy, reward_train_log, reward_valid_log, debugging_log =\
                train(
                    dataset_train, dataset_valid, policy, model_forward, optimizer,
                    train_cfg=train_cfg, search_cfg=search_cfg,
                    device=DEVICE, node_feat_cfg=args['dataset']['node_feat_cfg'],
                    edge_feat_cfg=args['dataset']['edge_feat_cfg']
                )

    ################# Plotting ###################
    num_iters_per_valid = args['inverse']['train_config']['num_iters_per_valid']
    num_iters_per_train = args['inverse']['train_config']['num_iters_per_train']
    print('plotting rewards')
    if reward_train_log is not None:
        plt.clf()
        plt.plot(torch.arange(len(reward_train_log)), reward_train_log, label='train')
        plt.plot(num_iters_per_valid*torch.arange(len(reward_valid_log)), reward_valid_log, label='valid')
        plt.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Reward')
        plt.savefig(os.path.join(log_dir, 'reward.png'))
    if debugging_log is not None:
        for k, v in debugging_log.items():
            plt.clf()
            plt.plot(num_iters_per_train*torch.arange(len(v)), v)
            plt.xlabel('Iterations')
            plt.ylabel(k)
            plt.savefig(os.path.join(log_dir, f'{k}.png'))

    ################# Kernel ###################
    g_train_best_li, curve_train_best_li = \
        fit_train_data_kernel(dataset_test, dlf_kernel, search_cfg)

    ################# Test ###################
    test_pn = os.path.join(log_dir,'test')
    reward, reward_min, reward_max, plot_obj = \
        test(
            dataset_test, dataset_train, policy, model_forward, search_cfg=search_cfg,
            device=DEVICE, skip_g_prog=False, num_runs=search_cfg.num_runs_test, g_train_best_li=g_train_best_li
        )
    graph_progress_li, graph_pred_li, g_dict_li, c_dict_li, c_dict_bg = plot_obj
    apply_train_kernel(dataset_test, dataset_kernel.dataset, g_train_best_li, curve_train_best_li, g_dict_li, c_dict_li)
    plot_obj = graph_progress_li, graph_pred_li, g_dict_li, c_dict_li, c_dict_bg
    print(f'reward:\t{reward:.3f}\t(min={reward_min:.3f}, max={reward_max:.3f})')

    ################# Plotting ###################
    if not os.path.isdir(test_pn):
        os.mkdir(test_pn)

    # plot_train_data = 'plot_train_data' in args and args['plot_train_data']
    # if plot_train_data:
    #     graph_progress_li, graph_pred_li, g_dict_li, c_dict_li, c_dict_bg = plot_obj
    #     cid2g_poly = fit_train_data_kernel(dataset_test, dlf_kernel, search_cfg, g_dict_li, c_dict_li, c_dict_bg)
    #     plot_obj = graph_progress_li, graph_pred_li, g_dict_li, c_dict_li, c_dict_bg

    plot_pred(*plot_obj, lim=(-1,1), dn=test_pn)

    ################# Exporting results ###################
    if 'export_results' in args and args['export_results']:
        graph_progress_li, graph_pred_li, g_dict_li, c_dict_li, *_ = plot_obj
        graph_true_li = [g_dict['True'] for g_dict in g_dict_li]
        graph_kernel_li = [g_dict['Train (0)'] for g_dict in g_dict_li]
        curve_pred_li = [c_dict['FwdModel(Pred Graph)'] for c_dict in c_dict_li]
        curve_tred_li = [c_dict['FwdModel(True Graph)'] for c_dict in c_dict_li]
        curve_true_li = [c_dict['True Curve'] for c_dict in c_dict_li]
        curve_kernel_li = [c_dict['Train Curve (0)[skip]'] for c_dict in c_dict_li]
        export_obj = \
            {
                'graph_progress_li':graph_progress_li,
                'graph_pred_li':graph_pred_li,
                'graph_true_li':graph_true_li,
                'graph_kernel_li':graph_kernel_li,
                'curve_RL_pred_li':curve_pred_li,
                'curve_pred_li':curve_tred_li,
                'curve_true_li':curve_true_li,
                'curve_kernel_li':curve_kernel_li
            }
        with open(os.path.join(log_dir, f'results.pkl'), "wb") as f:
            pickle.dump(export_obj, f)

    print(reward_train_log)
    # print(reward_valid_log)
    print(f'Time taken: {time.time()-t0}s')

def fit_train_data_kernel(dataset_test, dlf, search_cfg, topk=5):#, g_dict_li, c_dict_li, c_dict_bg, topk=5):
    print('running fit train data kernel')
    dlf.batch_size = 1
    dataset_train = dlf.get_train_dataset()

    # for i, (graph_train, curve_train) in enumerate(dataset_train):
    #     c_true_li, _ = \
    #         dataset_test.dataset.unnormalize_curve(curve_train.c_magnitude, curve_train.c_shape)
    #     for c_true in c_true_li:
    #         c_dict_bg[f'[skip]{i}'] = (c_true, None, None)

    g_train_best_li, curve_train_best_li = [], []
    # k = 0
    for i, (_, curve_test) in enumerate(dataset_test):
        print(f'batch {i} of {len(dataset_test)}')
        g_train_best = [[None for _ in range(len(curve_test))] for _ in range(topk)]
        curve_train_best = [None for _ in range(topk)]
        reward_cumsum_topk = float('-inf') * torch.ones(topk, len(curve_test))
        for graph_train, curve_train in tqdm(dataset_train): # only curve_true
            curve_train.c_li = curve_train.c_li * len(curve_test)
            curve_train.c_magnitude = curve_train.c_magnitude.repeat(len(curve_test), 1)
            curve_train.c_shape = curve_train.c_shape.repeat(len(curve_test), 1, 1)

            device = curve_train.c_shape.device
            model_surrogate_prediction = \
                (curve_train.c_shape, curve_train.c_magnitude,
                 torch.zeros_like(curve_train.c_shape, dtype=torch.float, device=device),
                 torch.zeros_like(curve_train.c_magnitude, dtype=torch.float, device=device))

            reward, _ = get_reward_helper(
                curve_test, model_surrogate_prediction, search_cfg, dataset_train,
                silent=True, skip_processing=True, PSU_specific_train_reward=False)
            curve_train = get_curve_helper(model_surrogate_prediction)

            indices_to_update = torch.ge(reward.to(reward_cumsum_topk.device), reward_cumsum_topk[-1])
            # if torch.isnan(reward).any():
            #     asdf = None
            #     reward, _ = get_reward_helper(
            #         curve_test, model_surrogate_prediction, search_cfg, dataset_train,
            #         silent=True, skip_processing=True)
            # print(reward)
            g_train_best[-1] = \
                [
                    graph_train.g_li[0] if update else old for (update, old) in
                    zip(to_list(indices_to_update), g_train_best[-1])
                ]
            curve_train_best[-1] = \
                curve_train.update(torch.logical_not(indices_to_update), curve_train_best[-1])
            reward_cumsum_topk[-1] = torch.maximum(reward.to(reward_cumsum_topk.device), reward_cumsum_topk[-1])

            reward_cumsum_topk, sorted_indices = torch.sort(reward_cumsum_topk, dim=0, descending=True)
            g_train_best_new = []
            for indices in sorted_indices:
                g_train_best_new.append([g_train_best[idx][j] for j, idx in enumerate(indices)])
            g_train_best = g_train_best_new
            curve_train_best_new = [deepcopy(curve_train_best[-1]) for _ in range(len(curve_train_best))]
            for j in range(len(curve_train_best_new)):
                for l in range(sorted_indices.shape[0]):
                    if curve_train_best[l] is not None:
                        curve_train_best_new[j].update(sorted_indices[j]==l, curve_train_best[l])
            curve_train_best = curve_train_best_new
        # g_train_best_li.extend(g_train_best)

        assert len(curve_train_best) == len(g_train_best)
        g_train_best_li.append(g_train_best)
        curve_train_best_li.append(curve_train_best)
        # for j in range(len(g_train_best[-1])):
        #     # for l in range(len(curve_train_best)):
        #     #     c_true, _ = \
        #     #         dataset_test.dataset.unnormalize_curve(curve_train_best[l].c_magnitude, curve_train_best[l].c_shape)
        #     #     g_dict_li[k][f'Train ({l})'] = g_train_best[l][j]
        #     #     c_dict_li[k][f'Train Curve ({l})[skip]'] = (c_true[j], None, None)
        #     # k += 1
    return g_train_best_li, curve_train_best_li

def apply_train_kernel(dataset_test, dataset_kernel, g_train_best_li, curve_train_best_li, g_dict_li, c_dict_li):
    k=0
    for i, ((_, curve_test), g_train_best, curve_train_best) in enumerate(
            zip(dataset_test, g_train_best_li, curve_train_best_li)):
        for j in range(len(g_train_best[-1])):
            for l in range(len(curve_train_best)):
                c_true, _ = \
                    dataset_test.dataset.unnormalize_curve(curve_train_best[l].c_magnitude, curve_train_best[l].c_shape)
                if 'digitize_cfg' in args['dataset'] and args['dataset']['digitize_cfg'] is not None:
                    c_kernel = dataset_kernel.get_unnormalized_curve_from_cid(g_train_best[l][j].graph['gid'])
                    # c_kernel_digital = \
                    #     dataset_kernel.digitize_curve_torch(
                    #         torch.tensor(c_kernel).unsqueeze(0)
                    #     ).squeeze(0)
                    # assert torch.all(c_kernel_digital == c_true[j].squeeze(-1)), \
                    #     f'{c_kernel_digital.cpu().numpy()} ... {c_true[j].squeeze(-1).cpu().numpy()}'
                else:
                    c_kernel = c_true[j]
                g_dict_li[k][f'Train ({l})'] = g_train_best[l][j]
                c_dict_li[k][f'Train Curve ({l})[skip]'] = (c_kernel, None, None)
            k += 1


def augment_dataset_RL(dataset_train):
    g_stats = dataset_train.dataset.g_stats
    c_stats = dataset_train.dataset.c_stats
    node_feats_index = dataset_train.dataset.g_stats
    edge_feats_index = dataset_train.dataset.edge_feats_index
    unnormalize_curve = dataset_train.dataset.unnormalize_curve
    args['dataset'].update(args['dataset_RL'])
    dlf = DataLoaderFactory(**args['dataset'])
    dlf.train_dataset.g_stats = g_stats
    dlf.train_dataset.c_stats = c_stats
    dataset_RL = dlf.get_train_dataset()
    combined_dataset = \
        torch.utils.data.ConcatDataset([
            dataset_train.dataset, dataset_RL.dataset])
    dataset_train = \
        torch.utils.data.DataLoader(
            dataset=combined_dataset,
            batch_size=dataset_train.batch_size,
            num_workers=dataset_train.num_workers,
            collate_fn=dataset_train.dataset.collate_fn,
            shuffle=True
        )
    dataset_train.dataset.g_stats = g_stats
    dataset_train.dataset.c_stats = c_stats
    dataset_train.dataset.node_feats_index = node_feats_index
    dataset_train.dataset.edge_feats_index = edge_feats_index
    dataset_train.dataset.unnormalize_curve = unnormalize_curve
    return dataset_train

def merge_dataset(dataset_train, dataset_RL):
    combined_dataset = \
        torch.utils.data.ConcatDataset([
            dataset_train.dataset, dataset_RL.dataset])
    dataset_train = \
        torch.utils.data.DataLoader(
            dataset=combined_dataset,
            batch_size=dataset_train.batch_size,
            num_workers=dataset_train.num_workers,
            collate_fn=dataset_train.dataset.collate_fn,
            shuffle=True
        )
    dataset_train.dataset.g_stats = dataset_RL.dataset.g_stats
    dataset_train.dataset.c_stats = dataset_RL.dataset.c_stats
    dataset_train.dataset.node_feats_index = dataset_RL.dataset.node_feats_index
    dataset_train.dataset.edge_feats_index = dataset_RL.dataset.edge_feats_index
    dataset_train.dataset.unnormalize_curve = dataset_RL.dataset.unnormalize_curve
    dataset_train.dataset.dim_input_nodes = dataset_RL.dataset.dim_input_nodes
    dataset_train.dataset.dim_input_edges = dataset_RL.dataset.dim_input_edges
    dataset_train.dataset.dim_curve = dataset_RL.dataset.dim_curve
    return dataset_train

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
    out = np.median(np.digitize(curve, bins).reshape(curve.shape[0],-1,blocksize),axis=-1).astype(int)
    # curve[:n_freq, 1] = \
    #     block_reduce(np.digitize(curve[:, 1], bins), block_size=blocksize, func=np.median)
    # curve[:n_freq, 0] = \
    #     block_reduce(curve[:, 0], block_size=blocksize, func=np.median)
    return out

if __name__ == '__main__':
    # with torch.autograd.set_detect_anomaly(False):
    main()
    # result_pn = '/home/derek/Documents/MaterialSynthesis/logs/benchmarking/inverse_transmission/imitation128/results.pkl'
    plt.clf()
    root_pn = log_dir
    # root_pn = '/home/derek/Documents/MaterialSynthesis/logs/benchmarking/inverse_stressstrain_FINAL/application_slopev2'
    # root_pn = '/home/derek/Documents/MaterialSynthesis/logs/benchmarking/inverse_stressstrain_FINAL/IL_slopev2'
    result_pn = f'{root_pn}/results.pkl'
    # result_pn = '/home/derek/Documents/MaterialSynthesis/logs/benchmarking/inverse_transmission/old_augmented512/results.pkl'
    # result_pn = '/home/derek/Downloads/results.pkl'
    with open(result_pn, 'rb') as fp:
        c = pickle.load(fp)
    if 'digitize_cfg' in args['dataset'] and args['dataset']['digitize_cfg'] is not None:
        c_train = torch.stack([
            torch.tensor(
                digitize_curve_np(args['dataset']['digitize_cfg'], np.expand_dims(x[0], axis=0))[0]
            ) for x in c['curve_kernel_li']], dim=0)
        c_true = torch.stack([
            torch.tensor(
                digitize_curve_np(args['dataset']['digitize_cfg'], np.expand_dims(x[0], axis=0))[0]
            ) for x in c['curve_true_li']], dim=0)
        c_pred = torch.stack([
            torch.tensor(
                digitize_curve_np(args['dataset']['digitize_cfg'], np.expand_dims(x[0], axis=0))[0]
            ) for x in c['curve_RL_pred_li']], dim=0)
    else:
        c_train = torch.stack([x[0].view(-1) for x in c['curve_kernel_li']], dim=0)
        c_true = torch.stack([torch.tensor(x[0]) for x in c['curve_true_li']], dim=0)
        c_pred = torch.stack([
            (torch.tensor(x[0]) if type(x[0]) == np.ndarray else x[0]).view(-1)
            for x in c['curve_RL_pred_li']], dim=0)

    if 'digitize_cfg' in args['dataset'] and args['dataset']['digitize_cfg'] is not None:
        diff_train = (c_train == c_true).float().mean(-1)
        diff_pred = (c_pred == c_true).float().mean(-1)
    else:
        diff_train = (torch.abs(c_train-c_true)/(torch.max(c_true, dim=-1)[0]-torch.min(c_true, dim=-1)[0]).unsqueeze(-1)).mean(dim=-1) # torch.abs(c_train-c_true).mean(dim=-1) #
        diff_pred = (torch.abs(c_pred-c_true)/(torch.max(c_true, dim=-1)[0]-torch.min(c_true, dim=-1)[0]).unsqueeze(-1)).mean(dim=-1) # torch.abs(c_pred-c_true).mean(dim=-1) #

    # plt.hist(diff_train[diff_train!=1.0], label=f'train, NMAE={diff_train[diff_train!=1.0].mean()}', alpha=0.5)
    # plt.hist(diff_pred[diff_train!=1.0], label=f'pred, NMAE={diff_pred[diff_train!=1.0].mean()}', alpha=0.5)
    plt.hist(diff_train, label=f'train, NMAE={diff_train.mean()}', alpha=0.5)
    plt.hist(diff_pred, label=f'pred, NMAE={diff_pred.mean()}', alpha=0.5)
    # plt.plot(c_true, label='true')
    plt.legend()
    plt.savefig(f'{root_pn}/tmp.png')
    plt.clf()
    plt.plot([np.argmax(x[0]) for x in c['curve_true_li']], diff_pred-diff_train)
    plt.savefig(f'{root_pn}/tmp_max.png')
    plt.clf()
    plt.plot([np.argmin(x[0]) for x in c['curve_true_li']], diff_pred-diff_train)
    plt.savefig(f'{root_pn}/tmp_min.png')
    plt.clf()