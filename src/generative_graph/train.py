# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 16:00:12 2023

@author: marco
"""
import os
import numpy as np
import math
import time
import torch
import random
import copy
import networkx as nx
import torch.nn.functional as F
from runstats import Statistics
from collections import deque, defaultdict
from torch_scatter import scatter_add
from copy import deepcopy

from src.utils import tboard
from src.config import args, log_dir
from src.generative_graph.env_v2 import run_action, run_environment_collated, StateCollated, ActionGraphObj, \
    sample_entropy
from src.generative_graph.test import test, run_search, graph2action_li

gamma = 1.0  # discount factor <--- not used
TOTAL_SEEN_SARP_PAIRS = 0
baseline_queue = deque([], 1_000_000)  # _000)
reward_stats = Statistics()


class SearchConfig:
    def __init__(
            self, magnitude_coeff, shape_coeff, jaccard_coeff, uncertainty_coeff,
            constraint_n_nodes_min, constraint_n_nodes_max,
            constraint_rho_min, constraint_rho_max, constraint_rho_default,
            num_runs_valid, num_runs_test, n_iterations_rho_binary_search
    ):
        # assert constraint_n_nodes_min > 2 # at least 3 nodes to satisfy face constraints
        assert constraint_n_nodes_max >= constraint_n_nodes_min
        assert constraint_rho_max >= constraint_rho_default >= constraint_rho_min
        self.magnitude_coeff = magnitude_coeff
        self.shape_coeff = shape_coeff
        self.jaccard_coeff = jaccard_coeff
        self.uncertainty_coeff = uncertainty_coeff
        self.constraint_n_nodes_min = constraint_n_nodes_min
        self.constraint_n_nodes_max = constraint_n_nodes_max
        self.constraint_rho_min = constraint_rho_min
        self.constraint_rho_max = constraint_rho_max
        self.constraint_rho_default = constraint_rho_default
        self.num_runs_valid = num_runs_valid
        self.num_runs_test = num_runs_test
        self.n_iterations_rho_binary_search = n_iterations_rho_binary_search


class TrainConfig:
    def __init__(self, num_iters, num_iters_per_train, num_iters_per_valid,
                 save_interval, clip_grad, scale_reward, norm_reward, clip_reward,
                 norm_adv, rl_algorithm, clip_coeff, entropy_coeff, use_scheduler,
                 num_imitation_epochs=0, **kwargs):
        self.num_iters = num_iters
        self.num_iters_per_train = num_iters_per_train
        self.num_iters_per_valid = num_iters_per_valid
        self.batch_size = None
        self.save_interval = save_interval
        self.norm_adv = norm_adv
        self.clip_grad = clip_grad
        assert rl_algorithm in ['REINFORCE', 'TRPO', 'PPO']
        self.rl_algorithm = rl_algorithm
        self.clip_coeff = clip_coeff
        self.entropy_coeff = entropy_coeff
        self.scale_reward = scale_reward
        self.norm_reward = norm_reward
        self.clip_reward = clip_reward
        self.use_scheduler = use_scheduler
        self.num_imitation_epochs = num_imitation_epochs


def pretrain_imitation(dataset_train, dataset_forward, policy, optimizer, train_cfg, search_cfg, device):
    loss_log = []
    for num_epoch in range(train_cfg.num_imitation_epochs):
        for graph_true, curve_true in dataset_train:
            probs = []
            graph_true.to(device), curve_true.to(device)
            action_li, start_li, end_li = graph2action_li(graph_true.g_li, device)

            rho_action = torch.tensor([g.graph['rho'] for g in graph_true.g_li], device=device)
            action_li.append(rho_action)

            policy_loss = torch.zeros(1, requires_grad=True, device=device, dtype=torch.float)

            *_, logs = \
                run_search(
                    curve_true=curve_true, graph_true=graph_true,
                    policy=policy, model_surrogate=None,
                    g_stats=dataset_forward.dataset.g_stats,
                    c_stats=dataset_forward.dataset.c_stats,
                    node_feats_index=dataset_train.dataset.node_feats_index,
                    edge_feats_index=dataset_train.dataset.edge_feats_index,
                    search_cfg=search_cfg,
                    dataset_forward=dataset_forward,
                    dataset_inverse=dataset_train,
                    device=device,
                    action_li=action_li, mcts_obj=None, n_samples_rho=1,
                    log_buffer=False, log_progress=False, log_logits=True, log_logprob=True)

            assert len(start_li) + 1 == len(end_li) + 1 == len(action_li) == len(logs['logits_li']) == len(
                logs['logprob_li'])
            weight = curve_true.weight_shape.to(device).view(-1)
            accum_loss = torch.tensor(0.0)

            for i, (logits, log_prob, action, start, end) in \
                    enumerate(zip(
                        logs['logits_li'][:-1], logs['logprob_li'][:-1],
                        action_li[:-1], start_li, end_li
                    )):
                loss_stop = \
                    (F.binary_cross_entropy_with_logits(
                        logits.logits_stop,
                        action.aid_stop.type(torch.float),
                        reduce=False
                    ).view(-1) * weight).mean()

                loss_start = torch.zeros(1, dtype=torch.float, device=logits.logits_aid_start.device)
                for aid_start in start:
                    mask = \
                        torch.logical_and(torch.ne(aid_start, -1),
                                          torch.logical_not(action.aid_stop).squeeze(-1))
                    if mask.any():
                        loss_start += \
                            (F.cross_entropy(
                                logits.logits_aid_start[mask].squeeze(-1),
                                aid_start[mask],
                                reduce=False
                            ).view(-1) * weight[mask]).mean()

                loss_end = torch.zeros(1, dtype=torch.float, device=logits.logits_aid_end.device)
                for aid_end in end:
                    mask = \
                        torch.logical_and(torch.ne(aid_end, -1),
                                          torch.logical_not(action.aid_stop).squeeze(-1))
                    if mask.any():
                        loss_end += \
                            (F.cross_entropy(
                                logits.logits_aid_end[mask].squeeze(-1),
                                aid_end[mask],
                                reduce=False
                            ).view(-1) * weight[mask]).mean()

                policy_loss = policy_loss + loss_stop + loss_start + loss_end

                if torch.logical_not(action.aid_stop).any():
                    probs.append(torch.exp(log_prob)[torch.logical_not(action.aid_stop).view(-1)].mean())

                # if num_epoch != train_cfg.num_imitation_epochs - 1:
                print(probs)
                policy_loss = torch.mean(policy_loss)
                accum_loss = accum_loss + policy_loss / len(start_li)
            # loss_rho = -5.0 * logs['logits_li'][-1].log_prob(action_li[-1]).mean()
            loss_rho = ((logs['logits_li'][-1].mean - action_li[-1]) ** 2).mean()
            accum_loss = accum_loss + loss_rho

            print(f'Loss: {accum_loss}')
            optimizer.zero_grad()
            accum_loss.backward()
            if train_cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), train_cfg.clip_grad)
            optimizer.step()
            loss_log.append(accum_loss.detach().cpu().item())
        print(f'{num_epoch} Epoch done!')

    return policy, loss_log


def plot_tsne_obj(tsne_obj):
    stop_token_time = np.concatenate(tsne_obj['stop_token_time'], axis=0).astype(int)
    start_token_class = np.concatenate(tsne_obj['start_token_class'], axis=0).astype(int)
    condition_emb = np.concatenate(tsne_obj['condition_emb'], axis=0)

    from sklearn.manifold import TSNE
    condition_emb = TSNE(n_components=2, learning_rate='auto').fit_transform(condition_emb)

    import matplotlib.pyplot as plt
    scatter = plt.scatter(condition_emb[:, 0], condition_emb[:, 1], c=start_token_class, s=5, cmap='tab20')
    handles, _ = scatter.legend_elements(prop='colors')
    plt.legend(handles, list(range(max(start_token_class))))
    plt.title('start token[0]')
    plt.savefig('/home/derek/Documents/MaterialSynthesis/src/generative_graph/_start.png')
    plt.clf()
    plt.close()

    scatter = plt.scatter(condition_emb[:, 0], condition_emb[:, 1], c=stop_token_time, s=5, cmap='tab20')
    handles, _ = scatter.legend_elements(prop='colors')
    plt.legend(handles, list(range(max(stop_token_time))))
    plt.title('time(stop token)')
    plt.savefig('/home/derek/Documents/MaterialSynthesis/src/generative_graph/_time.png')
    plt.clf()
    plt.close()
    # assert False


def update_tsne_obj(cur_state, action_li, policy, tsne_obj):
    # get stop token time
    stop_token_time = 999 * torch.ones(len(cur_state))
    for k, action in enumerate(action_li[:-1]):
        aid_stop = action.aid_stop.squeeze(-1)
        if True in aid_stop:
            stop_token_time[aid_stop] = \
                torch.minimum(
                    k * torch.ones_like(stop_token_time[aid_stop]),
                    stop_token_time[aid_stop])
    # USE_DEGREE_ORDERING
    start_token_class = action_li[0].aid_start
    # get embedding
    condition_emb = policy.get_embeddings(cur_state)['condition_emb']

    tsne_obj['stop_token_time'].append(stop_token_time.detach().cpu().numpy())
    tsne_obj['start_token_class'].append(start_token_class.detach().cpu().numpy())
    tsne_obj['condition_emb'].append(condition_emb.detach().cpu().numpy())


def train(dataset_train, dataset_valid, dataset_forward, policy, model_surrogate, optimizer,
          train_cfg, search_cfg, device, node_feat_cfg, edge_feat_cfg):
    global baseline_queue, TOTAL_SEEN_SARP_PAIRS
    model_surrogate.eval()

    # Optimizer
    reward_train_log = []
    reward_valid_log = []
    debugging_log = {
        'loss': [],
        'grad': [],
        'reward_cumsum': [],
        'reward_scaled': [],
        'reward_norm': [],
        'reward_clipped': []
    }
    policy_best = deepcopy(policy)
    reward_best, reward_min, reward_max, _ = \
        test(
            dataset_valid, dataset_forward, policy, model_surrogate, search_cfg=search_cfg, device=device,
            skip_g_prog=True, num_runs=search_cfg.num_runs_valid
        )
    print(f'untrained model\treward={reward_best:.3f}\t(min={reward_min:.3f}, max={reward_max:.3f})')
    cur_epoch, cur_iter, buffer = 0, 0, defaultdict(list)
    while cur_iter < train_cfg.num_iters:
        print(f'EPOCH: {cur_epoch} {cur_iter}')
        cur_epoch += 1
        for graph_true, curve_true in dataset_train:  # only curve_true
            if cur_iter >= train_cfg.num_iters:
                break
            graph_true.to(device), curve_true.to(device)
            cur_iter += 1

            with torch.no_grad():
                reward_cumsum, *_, logs = \
                    run_search(
                        curve_true=curve_true, graph_true=graph_true,
                        policy=policy, model_surrogate=model_surrogate,
                        g_stats=dataset_forward.dataset.g_stats,
                        c_stats=dataset_forward.dataset.c_stats,
                        node_feats_index=dataset_train.dataset.node_feats_index,
                        edge_feats_index=dataset_train.dataset.edge_feats_index,
                        search_cfg=search_cfg,
                        dataset_forward=dataset_forward,
                        dataset_inverse=dataset_train,
                        device=device,
                        action_li=None, mcts_obj=None, n_samples_rho=1,
                        log_buffer=True, log_progress=False, log_logits=False, log_logprob=False,
                        PSU_specific_train_reward=True)
            buffer_search = logs['buffer']
            reward_train_log.append(float(reward_cumsum.mean()))
            tboard.add_scalars(
                'reward/reward_cumsum',
                {'min': reward_cumsum.min(), 'max': reward_cumsum.max(), 'mean': reward_cumsum.mean()},
                cur_iter
            )
            debugging_log['reward_cumsum'].append(reward_cumsum.mean().item())

            if train_cfg.scale_reward:
                for r in reward_cumsum.detach().cpu().tolist():
                    reward_stats.push(r)
                reward_cumsum = reward_cumsum / (reward_stats.stddev() + 1e-12)
            tboard.add_scalars(
                'reward/reward_scaled',
                {'min': reward_cumsum.min(), 'max': reward_cumsum.max(), 'mean': reward_cumsum.mean()},
                cur_iter
            )
            debugging_log['reward_scaled'].append(reward_cumsum.mean().item())

            if train_cfg.norm_reward:
                for r in reward_cumsum.detach().cpu().tolist():
                    reward_stats.push(r)
                reward_cumsum = (reward_cumsum - reward_stats.mean()) / (reward_stats.stddev() + 1e-12)
            tboard.add_scalars(
                'reward/reward_norm',
                {'min': reward_cumsum.min(), 'max': reward_cumsum.max(), 'mean': reward_cumsum.mean()},
                cur_iter
            )
            debugging_log['reward_norm'].append(reward_cumsum.mean().item())

            if train_cfg.clip_reward is not None:
                reward_cumsum = torch.clip(
                    reward_cumsum,
                    min=train_cfg.clip_reward['min'],
                    max=train_cfg.clip_reward['max']
                )
            tboard.add_scalars(
                'reward/reward_clipped',
                {'min': reward_cumsum.min(), 'max': reward_cumsum.max(), 'mean': reward_cumsum.mean()},
                cur_iter
            )
            debugging_log['reward_clipped'].append(reward_cumsum.mean().item())
            #
            buffer = update_buffer(buffer, buffer_search, reward_cumsum)

            if cur_iter % train_cfg.num_iters_per_valid == 0:
                reward, *_ = \
                    test(
                        dataset_valid, dataset_forward, policy, model_surrogate,
                        device=device, search_cfg=search_cfg,
                        skip_g_prog=True, num_runs=search_cfg.num_runs_valid
                    )
                reward_valid_log.append(reward)
                print(f'{cur_iter}: validate model')
                if reward > reward_best:
                    print(f'{cur_iter}: new best model, reward={reward}')
                    policy_best = deepcopy(policy)
                    reward_best = reward
                else:
                    print(f'{cur_iter}: model did not improve, reward={reward}')

            if cur_iter % train_cfg.num_iters_per_train == 0:
                # compute gradient size
                grad_size = 0.0
                for p in policy.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        grad_size += param_norm.item() ** 2
                grad_size = float(grad_size ** (1. / 2))

                policy, loss_update = update_policy(
                    cur_iter, policy, buffer, optimizer,
                    train_cfg, search_cfg, device)
                loss = float(loss_update.mean())
                print(f'{cur_iter}: update policy')
                print(f'Loss: {loss}')

                debugging_log['loss'].append(loss)
                debugging_log['grad'].append(grad_size)
                buffer = defaultdict(list)

            # Save model
            dn_model = os.path.join(log_dir, 'best_models')
            if not os.path.isdir(dn_model):
                os.mkdir(dn_model)
            if cur_iter in [0, 1, 2, 4, 8, 16, 32, 64, 128] or cur_iter % 128 == 0:  # % train_cfg.save_interval == 0:
                write_model(policy, dn_model, cur_iter)

    return policy_best, reward_train_log, reward_valid_log, debugging_log


def update_buffer(buffer, buffer_search, reward_cumsum):
    for x in buffer_search['state']: x.to('cpu')
    for x in buffer_search['action']: x.to('cpu')
    for x in buffer_search['log_prob']: x.to('cpu') if x is not None else None
    buffer['log_prob'].extend(buffer_search['log_prob'])
    buffer['reward'].extend([reward_cumsum] * len(buffer_search['log_prob']))
    buffer['state'].extend(buffer_search['state'])
    buffer['action'].extend(buffer_search['action'])
    buffer['action_type'].extend(buffer_search['action_type'])
    buffer['weight'].extend(buffer_search['weight'])
    assert len(buffer['log_prob']) == len(buffer['reward']) \
           == len(buffer['state']) == len(buffer['action'])
    return buffer


def update_policy(cur_iter, policy, buffer, optimizer, train_cfg, search_cfg, device):
    global TOTAL_SEEN_SARP_PAIRS
    loss_li = []
    assert len(buffer['log_prob']) == len(buffer['reward']) == len(buffer['state']) == \
           len(buffer['action']) == len(buffer['action_type']) == len(buffer['weight'])

    for i, (log_prob, reward, state, action, action_type, weight) in \
            enumerate(zip(buffer['log_prob'], buffer['reward'], buffer['state'],
                          buffer['action'], buffer['action_type'], buffer['weight'])):
        state.to(device)
        action.to(device)

        reward_li = reward.detach().tolist()
        TOTAL_SEEN_SARP_PAIRS += len(reward_li)
        baseline_queue.extend(reward_li)
        print('baseline_queue:', len(baseline_queue))
        print('TOTAL_SEEN_SARP_PAIRS:', TOTAL_SEEN_SARP_PAIRS)
        baseline = np.mean(np.array(baseline_queue))
        advantage = reward - baseline

        tboard.add_scalars(
            'loss/reward',
            {'reward': reward.mean(), 'baseline': baseline},
            cur_iter + i
        )
        tboard.add_scalars(
            'loss/advantage',
            {'min': advantage.min(), 'max': advantage.max(), 'mean': advantage.mean()},
            cur_iter + i
        )

        # advantage = torch.tensor(advantage)
        if train_cfg.norm_adv:
            eps = np.finfo(np.float32).eps.item()
            advantage = (advantage - advantage.mean()) / (torch.std(advantage) + eps).detach()
        advantage = advantage.to(torch.device(device))
        tboard.add_scalars(
            'loss/advantage_norm',
            {'min': advantage.min(), 'max': advantage.max(), 'mean': advantage.mean()},
            cur_iter + i
        )

        _, _, log_prob_new, logits_new, _ = run_action(state, policy, search_cfg, action=action)
        log_prob_new = torch.clamp(log_prob_new, min=-3.0, max=3.0)
        log_prob = torch.clamp(log_prob, min=-3.0, max=3.0)
        entropy = sample_entropy(action_type, logits_new, action)
        ratio = \
            torch.exp(log_prob_new - log_prob.to(device).detach()) \
                if train_cfg.rl_algorithm in ['TRPO', 'PPO'] \
                else log_prob_new
        tboard.add_scalar('loss/log_prob_new', log_prob_new.mean(), cur_iter + i)
        tboard.add_scalar('loss/log_prob', log_prob.mean(), cur_iter + i)
        tboard.add_scalar('loss/log_prob_new-log_prob', (log_prob_new - log_prob).mean(), cur_iter + i)
        tboard.add_scalar('loss/advantage', advantage.mean(), cur_iter + i)
        tboard.add_scalar('loss/ratio', ratio.mean(), cur_iter + i)

        if train_cfg.rl_algorithm == 'PPO':
            assert train_cfg.clip_coeff is not None
            policy_loss_pt1 = \
                -advantage.view(1, -1) * ratio.view(1, -1)
            policy_loss_pt2 = \
                -advantage.view(1, -1) * torch.clamp(
                    ratio.view(1, -1), 1 - train_cfg.clip_coeff, 1 + train_cfg.clip_coeff
                )
            policy_loss = torch.max(policy_loss_pt1, policy_loss_pt2)
            tboard.add_scalar('loss/mask_percentage',
                              1.0 - torch.logical_and(
                                  (ratio > 1 - train_cfg.clip_coeff),
                                  (ratio < 1 + train_cfg.clip_coeff)
                              ).float().mean(), cur_iter + i)
            tboard.add_scalar('loss/policy_loss_pt1', policy_loss_pt1.mean(), cur_iter + i)
        else:
            policy_loss = \
                -advantage.view(1, -1) * ratio.view(1, -1)
        policy_loss = torch.mean(policy_loss.view(-1))  # * weight.to(policy_loss.device).view(-1))
        tboard.add_scalar('loss/policy_loss', policy_loss.mean(), cur_iter + i)
        policy_loss = policy_loss + train_cfg.entropy_coeff * entropy
        # policy_loss = train_cfg.entropy_coeff * entropy
        tboard.add_scalar('loss/entropy', entropy.mean(), cur_iter + i)
        tboard.add_scalar('loss/loss', policy_loss.mean(), cur_iter + i)

        # print(f'Iteration {cur_iter}: loss={policy_loss}')
        if not torch.isnan(policy_loss):
            if train_cfg.use_scheduler is not None or args['optimizer_decay'] is not None:
                optimizer.optimizer.zero_grad()
            else:
                optimizer.zero_grad()
            policy_loss.backward()
            # policy_loss.backward()
            if train_cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), train_cfg.clip_grad)
            optimizer.step()
        else:
            print(f'Encountered large loss: {policy_loss}')

        # Store training loss
        loss_li.append(float(policy_loss.detach().cpu()))
    loss_update = np.array(loss_li)
    return policy, loss_update


def write_model(model, dn, run_name):
    torch.save(model.state_dict(), os.path.join(dn, f'model_{run_name}.pt'))