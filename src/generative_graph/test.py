# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 16:00:12 2023

@author: marco
"""
import os
import numpy as np
import math
import torch
import pickle
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from copy import deepcopy
from tqdm import tqdm

from src.utils import plot_curve, CObj
from src.generative_curve.test import get_shape, get_metrics
from src.generative_graph.env_v2 import run_action, run_environment_collated, StateCollated, sample_entropy
from src.generative_graph.tesellate import tesselate
from src.utils import plot_3d, rho2r, to_list, plot_3d_wrapper
from src.dataset_preprocessing_collated import DATASET_CFG

N_SAMPLES_TEST = 8
rewards_li = deque([0.0], 100_000)
gamma = 1.0 # discount factor

def plot_pred(graph_progress_li, graph_pred_li, g_dict_li, c_dict_li, c_dict_bg, lim=None, dn=None, max_figs=500):
    assert len(graph_progress_li) == len(graph_pred_li) == len(g_dict_li) == len(c_dict_li)
    for i, (graph_pred, g_dict, c_dict) in \
            enumerate(tqdm(zip(graph_pred_li, g_dict_li, c_dict_li))):
        g_dict['Pred'] = graph_pred
        c_dict.update(c_dict_bg)
        plot_3d_wrapper(g_dict=g_dict, c_dict=c_dict, pn=os.path.join(dn, f'{i}_results.png'))
        if i > max_figs:
            break

    print('saving generated graphs...')
    for gid, graph_pred in enumerate(tqdm(graph_pred_li)):
        with open(os.path.join(dn, f'graph_{gid}.gpkl'), 'wb') as f:
            pickle.dump(graph_pred, f)

    print('plotting generated graphs...')
    for i, graph_progress in enumerate(tqdm(graph_progress_li)):
        fig = plt.figure(figsize=(45,5))
        max_len = min(9, len(graph_progress))
        for j, graph in enumerate(graph_progress[:max_len]):
            ax = fig.add_subplot(int(f'1{max_len}{j+1}'), projection="3d")
            ax.set_title(f'Step {j}/{len(graph_progress)}')
            plot_3d(graph, lim=lim, ax=ax)
        plt.savefig(os.path.join(dn, f'{i}_process.png'))
        plt.clf()
        plt.close()
        if i > max_figs:
            break


def test(dataset_test, dataset_forward, policy, model_surrogate, search_cfg, num_runs, device, skip_g_prog, g_train_best_li=None, **kwargs):
    tnsr2np = lambda x: x.view(-1).detach().cpu().numpy()
    global rewards_li
    policy.eval()
    model_surrogate.eval()

    reward_log = []
    cur_iter = 0

    # plot object
    graph_progress_li, graph_pred_li, g_dict_li, c_dict_li, c_dict_bg = \
        [], [], [], [], {}

    for bid, (graph_true, curve_true) in enumerate(dataset_test):  # only curve_true
        graph_true.to(device), curve_true.to(device)
        c_shape_true, c_magnitude_true = curve_true.c_shape, curve_true.c_magnitude
        c_shape_tred, c_magnitude_tred, c_shape_tred_std, c_magnitude_tred_std = \
            model_surrogate.inference(graph_true)

        cur_iter += 1
        graph_kernel_true = None
        with torch.no_grad():
            reward_cumsum_best, graph_pred_best, curve_pred_best, progress_best = \
                run_search_mcts(
                    kwargs_search={
                        'curve_true':curve_true,
                        'graph_true':graph_true,
                        'policy':policy,
                        'model_surrogate':model_surrogate,
                        'g_stats':dataset_forward.dataset.g_stats,
                        'c_stats':dataset_forward.dataset.c_stats,
                        'node_feats_index':dataset_test.dataset.node_feats_index,
                        'edge_feats_index':dataset_test.dataset.edge_feats_index,
                        'search_cfg':search_cfg,
                        'dataset_forward': dataset_forward,
                        'dataset_inverse':dataset_test,
                        'device':device,
                        'n_samples_rho':16
                    },
                    num_runs=num_runs,
                    graph_kernel_true=graph_kernel_true #graph_true.g_li
                )

        # update graph_li and c_li
        reward_log.extend(to_list(reward_cumsum_best))
        graph_progress_li.extend(progress_best)
        graph_pred_li.extend(graph_pred_best)
        c_magnitude_pred = curve_pred_best.c_magnitude
        c_shape_pred = curve_pred_best.c_shape
        c_magnitude_pred_std = curve_pred_best.c_magnitude_std
        c_shape_pred_std = curve_pred_best.c_shape_std
        c_tred_li, c_tred_UBLB = \
            dataset_forward.dataset.unnormalize_curve(
                c_magnitude_tred, c_shape_tred,
                c_magnitude_u=c_magnitude_tred_std,
                c_shape_u=c_shape_tred_std)
        c_pred_li, c_pred_UBLB = \
            dataset_forward.dataset.unnormalize_curve(
                c_magnitude_pred, c_shape_pred,
                c_magnitude_u=c_magnitude_pred_std,
                c_shape_u=c_shape_pred_std)
        c_true_li, _ = \
            dataset_forward.dataset.unnormalize_curve(c_magnitude_true, c_shape_true)

        for i, (g, c_pred, c_tred, c_true) in \
                enumerate(zip(graph_true.g_li, c_pred_li, c_tred_li, c_true_li)):
            if 'digitize_cfg' in args['dataset'] and args['dataset']['digitize_cfg'] is not None:
                c_true_dataset = tnsr2np(c_true)
                c_true = dataset_forward.dataset.get_unnormalized_curve_from_cid(g.graph['gid'])
                assert np.all(dataset_forward.dataset.digitize_curve_np(np.expand_dims(c_true, axis=0))[0] == c_true_dataset), \
                    print(dataset_forward.dataset.digitize_curve_np(np.expand_dims(c_true, axis=0))[0], c_true_dataset)
            else:
                c_true = tnsr2np(c_true)
            g_dict_li.append({'True': g})
            c_dict_li.append({
                'FwdModel(True Graph)': (
                    tnsr2np(c_tred),
                    tnsr2np(c_tred_UBLB[0][i]) if c_tred_UBLB is not None else None,
                    tnsr2np(c_tred_UBLB[1][i]) if c_tred_UBLB is not None else None),
                'FwdModel(Pred Graph)': (
                    tnsr2np(c_pred),
                    tnsr2np(c_pred_UBLB[0][i]) if c_pred_UBLB is not None else None,
                    tnsr2np(c_pred_UBLB[1][i]) if c_pred_UBLB is not None else None),
                'True Curve': (c_true, None,None),
            })

    plot_obj = (graph_progress_li, graph_pred_li, g_dict_li, c_dict_li, c_dict_bg)
    policy.train()
    return np.array(reward_log).mean(), np.array(reward_log).min(), np.array(reward_log).max(), plot_obj


import networkx as nx
import random
from src.generative_graph.env_v2 import run_action, run_environment_collated, StateCollated, ActionGraphObj
from src.dataset import ACTION_SPACE, ACTION_CONSTRAINTS
from src.config import args
USE_DEGREE_ORDERING = True
def graph2action_li(g_li, device):
    polyhedron_li = [g.graph['polyhedron'] for g in g_li]
    action_li_batch = []
    # edge_li_max = max([len(p.edges()) for p in polyhedron_li]) + 1
    action_start_li = []
    action_end_li = []
    for polyhedron in polyhedron_li:
        nid_li = sorted(list(polyhedron.nodes))
        assert nid_li == list(range(len(nid_li)))
        polyhedron_coords = torch.tensor(np.array([polyhedron.nodes[nid]['coord'] for nid in nid_li])) # TODOTODOTODOTODO
        aid_li = \
            ((polyhedron_coords.unsqueeze(0) - ACTION_SPACE.unsqueeze(1)) ** 2).sum(dim=-1).argmin(dim=0)

        if USE_DEGREE_ORDERING:
            # assert len(nid_li) == len(aid_li)
            score_li = [
                nx.degree(polyhedron, nid)*(ACTION_CONSTRAINTS.shape[1]+1) + sum(ACTION_CONSTRAINTS[aid])
                for (nid, aid) in zip(nid_li, aid_li)
            ]
            assert len(nid_li) == len(aid_li) == len(score_li)
            start = nid_li[max(list(range(len(nid_li))), key=lambda x: score_li[x])] #
        else:
            start = random.sample(nid_li, k=1)[0]

        eid_li = list(nx.bfs_edges(polyhedron, start))

        if USE_DEGREE_ORDERING:
            action_start = [[aid_li[u].item()] for (u, _) in eid_li]
            action_end = [[aid_li[v].item()] for (_, v) in eid_li]
        else:
            action_start = []
            remaining_start_node = set()
            for (u, v) in eid_li[::-1]:
                remaining_start_node.update({aid_li[u].item(), aid_li[v].item()})
                action_start = [list(deepcopy(remaining_start_node))] + action_start
            action_end = [[aid_li[n] for n in nx.neighbors(polyhedron, u)] for (u, _) in eid_li]

        action_start_li.append(action_start)
        action_end_li.append(action_end)
        action_li = \
            [[0, aid_li[u], aid_li[v]] for (u, v) in eid_li]
        action_li = action_li + [[1, 1, 1]]
        action_li_batch.append(action_li)

    action_li_len_max = max(len(action_li) for action_li in action_li_batch)
    constraint_n_nodes_max = args['inverse']['search']['constraint_n_nodes_max']
    start_li = \
        -torch.ones(
            len(action_li_batch), action_li_len_max, constraint_n_nodes_max,
            dtype=torch.long, device=device)
    end_li = \
        -torch.ones(
            len(action_li_batch), action_li_len_max, constraint_n_nodes_max,
            dtype=torch.long, device=device)
    action_li_t = \
        torch.ones(len(action_li_batch), action_li_len_max, 3, dtype=torch.long, device=device)
    for i in range(len(action_li_batch)):
        for j in range(len(action_start_li[i])):
            start_li[i, j, :len(action_start_li[i][j])] = torch.tensor(action_start_li[i][j])
        for j in range(len(action_end_li[i])):
            end_li[i, j, :len(action_end_li[i][j])] = torch.tensor(action_end_li[i][j])
        action_li_t[i, :len(action_li_batch[i])] = torch.tensor(action_li_batch[i])

    # for i, action_li in enumerate(action_li_batch):
    #     action_li_batch[i] += [[1,1,1]] * (action_li_len_max - len(action_li))
    # action_li_batch = torch.tensor(action_li_batch, device=device).permute(1,2,0)

    action_li = []
    for (aid_stop, aid_start, aid_end) in action_li_t.permute(1,2,0):# action_li_batch:
        ActionGraphObj(aid_stop, aid_start, aid_end)
        action_li.append(ActionGraphObj(aid_stop.view(-1,1).type(torch.bool), aid_start, aid_end))

    return action_li, start_li.permute(1,2,0), end_li.permute(1,2,0)

def run_search_sampling(kwargs_search, num_runs=8):
    progress_best, graph_pred_best, curve_pred_best = \
        [None for _ in range(len(kwargs_search['curve_true']))], \
        [None for _ in range(len(kwargs_search['curve_true']))], \
        None
    reward_cumsum_best = float('-inf') * torch.ones(len(kwargs_search['curve_true']), device=kwargs_search['device'])
    for _ in tqdm(range(num_runs)):
        reward, graph, curve, logs = \
            run_search(
                **kwargs_search, action_li=None, mcts_obj=None,
                log_buffer=False, log_progress=True, log_logits=False, log_logprob=False)

        indices_to_update = torch.ge(reward, reward_cumsum_best)
        if torch.mean(reward) > torch.mean(reward_cumsum_best):
            progress_best = \
                [
                    new if update else old for (update, old, new) in
                    zip(to_list(indices_to_update), progress_best, logs['progress_li'])
                ]
            graph_pred_best = \
                [
                    new if update else old for (update, old, new) in
                    zip(to_list(indices_to_update), graph_pred_best, graph.g_li)
                ]
            curve_pred_best = \
                curve.update(torch.logical_not(indices_to_update), curve_pred_best)
            reward_cumsum_best = torch.maximum(reward, reward_cumsum_best)
    assert None not in progress_best
    assert None not in graph_pred_best
    assert curve_pred_best is not None
    return reward_cumsum_best, graph_pred_best, curve_pred_best, progress_best

def run_search_mcts(kwargs_search, num_runs, graph_kernel_true, cpuct=2.5):
    progress_best, graph_pred_best, curve_pred_best = \
        [None for _ in range(len(kwargs_search['curve_true']))], \
        [None for _ in range(len(kwargs_search['curve_true']))], \
        None
    mcts_obj = MCTSObj(bs=len(kwargs_search['curve_true']), cpuct=cpuct) # TODO: subtract mean?
    reward_cumsum_best = float('-inf') * torch.ones(len(kwargs_search['curve_true']), device=kwargs_search['device'])
    for i in tqdm(range(num_runs)):
        mcts_obj.init_search()
        if False:# i == 0 and graph_kernel_true is not None:
            action_li, _, _ = graph2action_li(graph_kernel_true, kwargs_search['device'])
            rho_action = torch.tensor([g.graph['rho'] for g in graph_kernel_true], device=kwargs_search['device'])
            action_li.append(rho_action.unsqueeze(-1))
        else:
            action_li = None
        reward, graph, curve, logs = \
            run_search(
                **kwargs_search, action_li=action_li, mcts_obj=mcts_obj,
                log_buffer=False, log_progress=True, log_logits=False, log_logprob=False)

        indices_to_update = torch.ge(reward, reward_cumsum_best)
        if torch.mean(reward) > torch.mean(reward_cumsum_best):
            progress_best = \
                [
                    new if update else old for (update, old, new) in
                    zip(to_list(indices_to_update), progress_best, logs['progress_li'])
                ]
            graph_pred_best = \
                [
                    new if update else old for (update, old, new) in
                    zip(to_list(indices_to_update), graph_pred_best, graph.g_li)
                ]
            curve_pred_best = \
                curve.update(torch.logical_not(indices_to_update), curve_pred_best)
            reward_cumsum_best = torch.maximum(reward, reward_cumsum_best)
    assert None not in progress_best
    assert None not in graph_pred_best
    assert curve_pred_best is not None
    return reward_cumsum_best, graph_pred_best, curve_pred_best, progress_best



class MCTSObj:
    def __init__(self, bs, cpuct):
        self.cpuct = cpuct
        self.bs = bs
        self.delimiter = '_-_'

        self.N = 0
        self.sa2N = defaultdict(lambda: 0)
        self.s2N = defaultdict(lambda: 0)
        self.sa2R = defaultdict(lambda: 0)
        self.baseline_li = [0 for _ in range(self.bs)]
        self.h_li = [[str(i)] for i in range(self.bs)]

    def init_search(self):
        self.h_li = [[str(i)] for i in range(self.bs)]

    def init_action(self):
        for i in range(len(self.h_li)):
            if self.h_li[i][-1] != 'True':
                assert self.h_li[i][-1] != 'False'
                self.h_li[i].append('False')

    def sample_action(self, probs, masks, is_stop_token=False, action=None):
        assert self.bs == probs.shape[0] == masks.shape[0] == len(self.h_li) == len(self.baseline_li)
        # masks/probs: Bx|A| (stop-token, |A|=2; u, |A|=22; v, |A|=22)

        out_li = []
        score_li = []
        for bid, (h, baseline, prob, mask) in enumerate(zip(self.h_li, self.baseline_li, probs, masks)): # looping through batch
            action_li = mask.nonzero().view(-1).detach().cpu().tolist() # indices of valid actions 1 x |validA|
            probs_li = prob[mask].detach().cpu().numpy() # probs of valid actions 1 x |validA|
            score = []
            for a, p in zip(action_li, probs_li):
                h_sa = self.delimiter.join(self._action2str(h, a, is_stop_token)) # hash function: s, a -> string
                Q = self.sa2R[h_sa]-baseline if h_sa in self.sa2R else 0.0 # <-- estimated reward from Alphago Zero
                N = self.sa2N[h_sa] if h_sa in self.sa2N else 0.0 # <-- visit count from Alphago Zero
                score.append(Q + self.cpuct * p/(1+N))
            score = np.array(score)
            if action is None:
                action_id = np.random.choice(np.flatnonzero(score == score.max()))
                out_li.append(action_li[action_id])
                # out_li.append(action_li[score.argmax()])
            else:
                out_li.append(action[bid])
            score_li.append(score)
        return out_li, score_li

    def commit_action(self, action, is_stop_token=False):
        assert len(action) == len(self.h_li)
        for i, (h, a) in enumerate(zip(self.h_li, action)):
            self.h_li[i] = self._action2str(h, a, is_stop_token)

    def commit_search(self, r):
        self.N += 1
        for i, h in enumerate(self.h_li):
            for j in range(len(h)):
                if j == 0 or h[j] == 'False':
                    continue
                h_subsample = h[:j+1]
                self.sa2N[self.delimiter.join(h_subsample)] += 1
                self.s2N[self.delimiter.join(h_subsample[:-1])] += 1

                m = self.sa2N[self.delimiter.join(h_subsample)]
                self.sa2R[self.delimiter.join(h_subsample)] = \
                    (m-1)/m*self.sa2R[self.delimiter.join(h_subsample)] + 1/m*r[i].item()
            self.baseline_li[i] = (self.N-1)/self.N * self.baseline_li[i] + 1/self.N * r[i].item()

    def _action2str(self, h, a, is_stop_token):
        if is_stop_token:
            if a == 1:
                h_sa = h[:-3] + ['True'] + h[-2:]
            else:
                h_sa = h
                assert h[-3] != 'True'
        else:
            h_sa = h + [str(a)]
        return h_sa

def run_search(
        curve_true, graph_true, policy, model_surrogate, g_stats, c_stats,
        node_feats_index, edge_feats_index, search_cfg, dataset_forward, dataset_inverse,
        device=None, action_li=None, mcts_obj=None, n_samples_rho=16,
        log_buffer=False, log_progress=False, log_logits=False, log_logprob=False, PSU_specific_train_reward=False):
    buffer = defaultdict(list)
    progress_li = []
    logits_li = []
    logprob_li = []

    num_iters = 0
    cur_state = StateCollated.init_initial_state(curve_true, g_stats, c_stats)
    reward_cumsum = torch.zeros(len(cur_state), device=device, dtype=torch.float)
    while True:
        if cur_state.done_rho:
            assert action_li is None or len(action_li) == num_iters
            break
        action = None if action_li is None else action_li[num_iters]

        action_type, action, log_prob, logits, mask_obj = \
            run_action(
                cur_state, policy, search_cfg,
                argmax=False, n_samples_rho=n_samples_rho,
                mcts_obj=mcts_obj, action=action)
        next_state, reward, graph, curve = \
            run_environment_collated(
                action_type, action, cur_state, model_surrogate, graph_true, g_stats,
                node_feats_index, edge_feats_index, search_cfg, dataset_forward, dataset_inverse,
                PSU_specific_train_reward=PSU_specific_train_reward, device=device)
        # entropy = sample_entropy(action_type, logits, action)
        reward_cumsum = reward_cumsum + reward

        if log_logprob:
            logprob_li.append(log_prob)
        if log_logits:
            logits_li.append(logits)
        if log_buffer and not cur_state.done_graph:
            if search_cfg.n_iterations_rho_binary_search is None or not cur_state.done_graph:
                buffer['state'].append(cur_state)
                buffer['action'].append(action)
                buffer['log_prob'].append(log_prob)
                buffer['action_type'].append(action_type)
                buffer['weight'].append(curve_true.weight_shape.view(-1))
        if log_progress:
            progress_li.append(get_graph_progress(next_state))
            # pass

        cur_state = next_state
        num_iters += 1

    if log_progress:
        progress_li = list(zip(*progress_li))
    logs = {
        'buffer':buffer,
        'progress_li':progress_li,
        'logits_li':logits_li,
        'logprob_li':logprob_li
    }
    return reward_cumsum, graph, curve, logs

def get_graph_progress(cur_state):
    tetrahedron_li = \
        cur_state.graph_collated.get_g_li(
            cur_state.graph_collated.feats_node,
            cur_state.graph_collated.edge_index,
            cur_state.graph_collated.graph_index,
            rho=None)
    g_li = []
    for tetrahedron in tetrahedron_li:
        g = tesselate(
            tetrahedron,
            start='tetrahedron',
            end=DATASET_CFG['base_cell'],
            rm_redundant=True
        )
        g_li.append(g)
    return g_li