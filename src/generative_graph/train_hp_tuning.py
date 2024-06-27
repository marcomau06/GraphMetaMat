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
import optuna
import networkx as nx
from runstats import Statistics
from collections import deque, defaultdict
from torch_scatter import scatter_add
from copy import deepcopy

from src.config import args, log_dir
from src.generative_graph.env_v2 import run_action
from src.generative_graph.test import test, run_search

gamma = 1.0 # discount factor <--- not used
baseline_queue = deque([], 1_000_000)
reward_stats = Statistics()

class SearchConfig:
    def __init__(
            self, magnitude_coeff, shape_coeff, jaccard_coeff, uncertainty_coeff,
            constraint_n_nodes_min, constraint_n_nodes_max,
            constraint_rho_min, constraint_rho_max, constraint_rho_default,
            num_runs_valid, num_runs_test, n_iterations
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
        self.n_iterations = n_iterations_rho_binary_search

class TrainConfig:
    def __init__(self, num_iters, num_iters_per_train, num_iters_per_valid,
                 save_interval, clip_grad, scale_reward, norm_reward, clip_reward,
                 norm_adv, rl_algorithm, clip_coeff, entropy_coeff, use_scheduler, **kwargs):
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

def train(dataset_train, dataset_valid, policy, model_surrogate, optimizer,
          train_cfg, search_cfg, trial, device, node_feat_cfg, edge_feat_cfg):
    global baseline_queue
    model_surrogate.eval()
    
    # Optimizer
    # reward_train_log = []
    # reward_valid_log = []
    # loss_log = []
    policy_best = deepcopy(policy)
    reward_best, reward_min, reward_max, _ = \
        test(
            dataset_valid, policy, model_surrogate, search_cfg=search_cfg, device=device,
            skip_g_prog=True, num_runs=search_cfg.num_runs_valid
        )
    print(f'untrained model\treward={reward_best:.3f}\t(min={reward_min:.3f}, max={reward_max:.3f})')
    cur_epoch, cur_iter, buffer = 0, 0, defaultdict(list)
    while cur_iter < train_cfg.num_iters:        
        print(f'EPOCH: {cur_epoch} {cur_iter}')
        cur_epoch += 1
        for graph_true, curve_true in dataset_train: # only curve_true
            if cur_iter >= train_cfg.num_iters:
                break
            graph_true.to(device), curve_true.to(device)
            cur_iter += 1
            buffer_search, reward_cumsum, *_ = \
                run_search(
                    curve_true, graph_true, policy, model_surrogate,
                    dataset_train.dataset.g_stats,
                    dataset_train.dataset.c_stats,
                    dataset_train.dataset.node_feats_index,
                    dataset_train.dataset.edge_feats_index,
                    search_cfg=search_cfg, dataset=dataset_train,
                    device=device, skip_g_prog=True)
            # reward_train_log.append(float(reward_cumsum.mean()))

            if train_cfg.scale_reward:
                for r in reward_cumsum.detach().cpu().tolist():
                    reward_stats.push(r)
                reward_cumsum = reward_cumsum / (reward_stats.stddev() + 1e-12)
            if train_cfg.norm_reward:
                for r in reward_cumsum.detach().cpu().tolist():
                    reward_stats.push(r)
                reward_cumsum = (reward_cumsum - reward_stats.mean()) / (reward_stats.stddev() + 1e-12)
            if train_cfg.clip_reward is not None:
                reward_cumsum = torch.clip(
                    reward_cumsum,
                    min=train_cfg.clip_reward['min'],
                    max=train_cfg.clip_reward['max']
                )
            buffer = update_buffer(buffer, buffer_search, reward_cumsum)

            if cur_iter % train_cfg.num_iters_per_valid == 0:
                reward, *_ = \
                    test(
                        dataset_valid, policy, model_surrogate,
                        device=device, search_cfg=search_cfg,
                        skip_g_prog=True, num_runs=search_cfg.num_runs_valid
                    )
                # reward_valid_log.append(reward)
                print(f'{cur_iter}: validate model')
                if reward > reward_best:
                    print(f'{cur_iter}: new best model, reward={reward}')
                    policy_best = deepcopy(policy)
                    reward_best = reward
                else:
                    print(f'{cur_iter}: model did not improve, reward={reward}')

            if cur_iter % train_cfg.num_iters_per_train == 0:
                policy, loss_update = update_policy(
                    policy, buffer, optimizer,
                    train_cfg, search_cfg, device)
                loss = float(loss_update.mean())
                print(f'{cur_iter}: update policy')
                print(f'Loss: {loss}')
                # loss_log.append(loss)
                buffer = defaultdict(list)
            
            # # Save model
            # dn_model = os.path.join(log_dir, 'best_models')
            # if not os.path.isdir(dn_model):
            #     os.mkdir(dn_model)
            # if cur_iter % train_cfg.save_interval == 0:
            #     write_model(policy, dn_model, cur_iter)
                
        trial.report(reward_best, cur_epoch)     
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()                
                
    return policy_best, reward_best

def update_buffer(buffer, buffer_search, reward_cumsum):
    for x in buffer_search['state']: x.to('cpu')
    for x in buffer_search['action']: x.to('cpu')
    for x in buffer_search['log_prob']: x.to('cpu')
    buffer['log_prob'].extend(buffer_search['log_prob'])
    buffer['reward'].extend([reward_cumsum] * len(buffer_search['log_prob']))
    buffer['state'].extend(buffer_search['state'])
    buffer['action'].extend(buffer_search['action'])
    buffer['entropy'].extend(buffer_search['entropy'])
    assert len(buffer['log_prob']) == len(buffer['reward']) \
           == len(buffer['state']) == len(buffer['action'])
    return buffer

def update_policy(policy, buffer, optimizer, train_cfg, search_cfg, device):
    loss_li = []
    for log_prob, reward, state, action, entropy in \
            zip(buffer['log_prob'], buffer['reward'], buffer['state'], buffer['action'], buffer['entropy']):
        state.to(device)
        action.to(device)

        baseline_queue.extend(reward.detach().tolist())
        baseline = np.mean(np.array(baseline_queue))
        advantage = reward - baseline
        # advantage = torch.tensor(advantage)
        if train_cfg.norm_adv:
            eps = np.finfo(np.float32).eps.item()
            advantage = (advantage - advantage.mean()) / (torch.std(advantage) + eps).detach()
        advantage = advantage.to(torch.device(device))

        _, _, log_prob_new, _, _ = run_action(state, policy, search_cfg, action=action)
        ratio = \
            log_prob_new - log_prob.to(device).detach() \
                if train_cfg.rl_algorithm in ['TRPO', 'PPO'] \
                else log_prob_new

        if train_cfg.rl_algorithm == 'PPO':
            assert train_cfg.clip_coeff is not None
            policy_loss_pt1 = \
                -advantage.view(1, -1) * ratio.view(1, -1)
            policy_loss_pt2 = \
                -advantage.view(1, -1) * torch.clamp(
                    ratio.view(1, -1), 1 - train_cfg.clip_coeff, 1 + train_cfg.clip_coeff
                )
            policy_loss = torch.max(policy_loss_pt1, policy_loss_pt2)
        else:
            policy_loss = \
                -advantage.view(1, -1) * ratio.view(1, -1)
        policy_loss = torch.mean(policy_loss)
        policy_loss = policy_loss + train_cfg.entropy_coeff * entropy

        # print(f'Iteration {cur_iter}: loss={policy_loss}')
        if not torch.isnan(policy_loss):
            if train_cfg.use_scheduler is not None:
                optimizer.optimizer.zero_grad()
            else:
                optimizer.zero_grad()
            policy_loss.backward()
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
