import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import random
from copy import deepcopy
from torch_scatter import scatter_add
from torch.distributions import Categorical

from src.dataset import GraphObjCollatedv2
from src.utils import plot_3d, PLUS_MINUS, CObj, to_item, rho3r
from src.dataset_preprocessing_collated import DATASET_CFG
from src.dataset import ACTION_SPACE, ACTION_CONSTRAINTS, TETRAHEDRON
from src.config import args, TASK
DEVICE = args['device']

NODES = 'nodes'
EDGES = 'edges'
RADIUS = 'radius'

ACTION_CONSTRAINTS = ACTION_CONSTRAINTS.to(DEVICE)
NUM_FACES_CONSTRAINTS = ACTION_CONSTRAINTS.shape[-1]
NUM_SIZE_CONSTRAINTS = 2
NUM_CONSTRAINTS = NUM_FACES_CONSTRAINTS + NUM_SIZE_CONSTRAINTS

ACTION_CONSTRAINTS_HASH = torch.pow(2, torch.arange(NUM_FACES_CONSTRAINTS)).view(-1,1)


##############################################################
# State and Action Definition
##############################################################

class LogitGraphObj:
    def __init__(self, logits_stop, logits_aid_start, logits_aid_end):
        self.logits_stop = logits_stop
        self.logits_aid_start = logits_aid_start
        self.logits_aid_end = logits_aid_end

    def to(self, device):
        self.logits_stop = self.logits_stop.to(device)
        self.logits_aid_start = self.logits_aid_start.to(device)
        self.logits_aid_end = self.logits_aid_end.to(device)

class ActionGraphObj:
    def __init__(self, aid_stop, aid_start, aid_end):
        self.aid_stop = aid_stop
        self.aid_start = aid_start
        self.aid_end = aid_end

    def to(self, device):
        self.aid_stop = self.aid_stop.to(device)
        self.aid_start = self.aid_start.to(device)
        self.aid_end = self.aid_end.to(device)

class StateCollated():
    def __init__(self, graph_collated, rho_collated, curve_collated,
                 constraints, done_graph, done_rho, g_stats, c_stats):
        self.graph_collated = graph_collated
        self.rho_collated = rho_collated
        self.curve_collated = curve_collated
        self.constraints = constraints
        self.done_graph = done_graph
        self.done_rho = done_rho
        self.g_stats = g_stats
        self.c_stats = c_stats

    def __len__(self):
        return len(self.curve_collated)

    def is_done_graph(self):
        return False not in self.constraints

    def is_done_rho(self):
        return self.rho_collated is not None

    def get_curve(self):
        return self.curve_collated

    def to(self, device):
        if self.graph_collated is not None:
            self.graph_collated.to(device)
        if self.curve_collated is not None:
            self.curve_collated.to(device)
        if self.rho_collated is not None:
            self.rho_collated = self.rho_collated.to(device)
        if self.constraints is not None:
            self.constraints = self.constraints.to(device)

    @classmethod
    def init_initial_state(cls, curve_collated, g_stats, c_stats):
        bs = len(curve_collated)
        return cls(
            None, None, curve_collated,
            torch.zeros(bs, NUM_CONSTRAINTS, dtype=torch.bool),
            False, False, g_stats, c_stats)

##############################################################
# Symmetry and Connectivity Constraints
##############################################################
def get_mask_face_token(state, constraints_unsat, n_faces_unsat, constraints_sat_all_faces=None, device='cpu'):
    # default: stop token entry is true
    graph = state.graph_collated
    if graph is None:
        # select any node on empty graph
        assert constraints_unsat is None
        assert n_faces_unsat is None
        mask_face_token = \
            torch.ones(
                len(state.curve_collated), ACTION_SPACE.shape[0],
                dtype=torch.bool, device=device) # b x n_act_nodes
    else:
        # select nodes that contain faces not in the current graph
        assert constraints_unsat is not None
        assert n_faces_unsat is not None
        if constraints_sat_all_faces is None:
            mask_face_token = \
                torch.matmul(
                    constraints_unsat.type(torch.float),
                    ACTION_CONSTRAINTS.to(graph.graph_index.device).transpose(0,1).type(torch.float)
                ) >= 1 # b x n_act_nodes
        else:
            constraints_n_faces_min = torch.ones_like(n_faces_unsat)
            constraints_n_faces_min[constraints_sat_all_faces] = \
                n_faces_unsat[constraints_sat_all_faces]
            assert torch.all(torch.le(constraints_n_faces_min, 2))
            constraints_n_faces_min = constraints_n_faces_min.view(-1, 1)

            mask_face_token = \
                torch.ge(torch.matmul(
                    constraints_unsat.type(torch.float),
                    ACTION_CONSTRAINTS.to(graph.graph_index.device).transpose(0,1).type(torch.float)
                ), constraints_n_faces_min) # b x n_act_nodes

        # find cases where all constraints are satisfied (i.e. no constraints are unsatisfied)
        constraints_all_sat = torch.eq(n_faces_unsat, 0) # bs x 1
        mask_face_token[constraints_all_sat] = True # b x n_act_nodes
    return mask_face_token

def get_mask_stop_token(state, search_cfg, num_nodes, n_faces_unsat, device='cpu'):
    assert search_cfg.constraint_n_nodes_max > search_cfg.constraint_n_nodes_min

    # don't select stop token on empty graph
    mask_stop_token = torch.zeros(len(state), 1, dtype=torch.bool, device=device) # b x 1
    mask_terminal = torch.zeros(len(state), 1, dtype=torch.bool, device=device) # b x 1

    # fill batch indices with non-empty graphs
    graph = state.graph_collated
    if graph is not None:
        assert num_nodes is not None
        mask_stop_token = \
            torch.unsqueeze(torch.ge(num_nodes, search_cfg.constraint_n_nodes_min), dim=-1)
        mask_stop_token = torch.logical_and(torch.eq(n_faces_unsat, 0).view(-1,1), mask_stop_token)
        mask_terminal = \
            torch.logical_and(
                mask_stop_token,
                torch.unsqueeze(
                    torch.ge(num_nodes, search_cfg.constraint_n_nodes_max), dim=-1))
    else:
        assert num_nodes is None

    return mask_stop_token, mask_terminal

def get_mask_unselected_aids(state, aid_selected, device='cpu'):
    graph = state.graph_collated
    if graph.aid_index is None:
        mask_unselected_aids = \
            torch.ones(
                len(state.curve_collated), ACTION_SPACE.shape[0],
                dtype=torch.bool, device=device) # b x n_act_nodes
    else:
        constraints_sel_u = \
            ACTION_CONSTRAINTS[
                aid_selected.view(-1)
            ].view(len(state.curve_collated), 1, NUM_FACES_CONSTRAINTS).to(device) # b x 1 x n_faces
        # default edge covers all faces which doesn't exist.
        constraints_sel_u_neighbors = \
            torch.zeros(
                len(state.curve_collated),
                ACTION_SPACE.shape[1],
                NUM_FACES_CONSTRAINTS,
                dtype=torch.bool, device=device
            ) # b x max(neighbors) x n_faces

        # collect the faces of all neighboring nodes
        for bid, aid_sel in enumerate(aid_selected):
            tetra_aid_li = set(list(TETRAHEDRON.neighbors(int(aid_sel))))
            graph_aid_li = set(graph.aid_index[graph.graph_index == bid].detach().cpu().tolist())
            # neighboring aid = intersection(tetrahedron neighbors, graph nodes)
            neighbor_aid_li = list(tetra_aid_li.intersection(graph_aid_li))
            constraints_sel_u_neighbors[bid, :len(neighbor_aid_li)] = ACTION_CONSTRAINTS[neighbor_aid_li]

        # collect faces of all neighboring edges
        edge_constraints = \
            torch.logical_and(
                constraints_sel_u.repeat(1, constraints_sel_u_neighbors.shape[1], 1),
                constraints_sel_u_neighbors
            ) # b x max(neighbors) x 4
        assert torch.all(torch.sum(edge_constraints.type(torch.float), dim=-1) < 3)

        collisions_edge_constraints = \
            torch.matmul(
                edge_constraints.type(torch.float),
                ACTION_CONSTRAINTS.type(torch.float).to(device).transpose(0,1)
            ) == 2 # b x max(neighbors) x n_nodes_act

        mask_unselected_aids = \
            torch.sum(collisions_edge_constraints.type(torch.float), dim=1) == 0
    return mask_unselected_aids

def get_mask_initial(state, constraints_n_new_nodes_max=None, device='cpu'):
    graph = state.graph_collated
    bs, num_actions = len(state.curve_collated), ACTION_SPACE.shape[0]
    if graph is None:
        mask_initial = \
            torch.ones(bs, num_actions, dtype=torch.bool, device=device) # b x n_act_nodes
    else:
        mask_initial = \
            torch.zeros(bs, num_actions, dtype=torch.bool, device=device) # b x n_act_nodes
        graph_index_base = graph.graph_index[graph.graph_base_index]
        graph_index_all = graph.graph_index
        aid_index_base = graph.aid_index[graph.graph_base_index]
        aid_index_all = graph.aid_index

        if constraints_n_new_nodes_max is None:
            mask_initial[graph_index_all, aid_index_all] = True
        else:
            assert torch.all(torch.ge(constraints_n_new_nodes_max, 0))

            # action adds 1 graph_base_index nodes
            mask_initial[graph_index_base, aid_index_base] = True

            # action adds 2 graph_base_index nodes for graph_index where cons_n_new_nodes_max > 1
            mask_initial[
                graph_index_all[constraints_n_new_nodes_max[graph_index_all] > 1],
                aid_index_all[constraints_n_new_nodes_max[graph_index_all] > 1]
            ] = True

    return mask_initial

def sample_action_aid_stop(logits_stop, mask_stop_token, mask_terminal, mcts_obj=None, action=None):
    probs = F.sigmoid(logits_stop)
    if mcts_obj is None:
        act = mask_stop_token * (probs > torch.rand_like(logits_stop))
        out = torch.logical_or(mask_terminal, act)
        assert action is None, 'if we know the action and mcts_obj is None, we should skip this function and return action'
    else:
        masks = torch.cat([torch.logical_not(mask_terminal), mask_stop_token], dim=-1)
        probs = torch.cat([1-probs, probs], dim=-1)
        out, _ = mcts_obj.sample_action(probs, masks, is_stop_token=True, action=action)
        mcts_obj.commit_action(out, is_stop_token=True)
        out = torch.tensor(out, dtype=torch.bool, device=probs.device)
    return out

def sample_action_aid_node(logits_aid, mask, mcts_obj, action=None, inject_noise=0.0):
    if inject_noise > 0.0:
        dummy = torch.ones_like(logits_aid, device=logits_aid.device)
        logits_aid += torch.normal(0.0*dummy, logits_aid.std()*inject_noise*dummy)

    probs = mask * (F.softmax(logits_aid.squeeze(-1), dim=-1)+1e-10)
    probs = probs/probs.sum(dim=-1).unsqueeze(-1)
    if mcts_obj is None:
        m = Categorical(probs)
        out = m.sample()
        assert action is None, 'if we know the action and mcts_obj is None, we should skip this function and return action'
    else:
        out, _ = mcts_obj.sample_action(probs, mask, is_stop_token=False, action=action)
        mcts_obj.commit_action(out, is_stop_token=False)
        out = torch.tensor(out, device=probs.device)
    return out

##############################################################
# Relative Densities Interface
##############################################################

def sample_action_rho(logits_rho, search_cfg, n_samples=1):
    rho = logits_rho.sample(sample_shape=torch.Size([n_samples])).transpose(0,1)
    rho = \
        torch.clamp(
            rho,
            min=search_cfg.constraint_rho_min,
            max=search_cfg.constraint_rho_max)
    # rho = logits_rho.mean.unsqueeze(-1).repeat(1,n_samples)
    return rho

def run_action_rho(state, policy, search_cfg, n_samples=1, action=None):
    embeddings = policy.get_embeddings(state)
    logits = policy.get_rho(state, **embeddings)
    if action is None:
        action = sample_action_rho(logits, search_cfg, n_samples=n_samples)
    return action, logits

##############################################################
# Action Interface
##############################################################

def run_action(state, policy, search_cfg, argmax=False, n_samples_rho=1, mcts_obj=None, action=None):
    if state.done_graph:
        action, logits = run_action_rho(state, policy, search_cfg=search_cfg, n_samples=n_samples_rho, action=action)
        mask_obj = None
        action_type = 'rho'
        if n_samples_rho == 1:
            log_prob = get_logprob(state, action, logits, mask_obj)
        else:
            log_prob = None
    else:
        action, logits, mask_obj = \
            run_action_graph(state, policy, search_cfg=search_cfg, mcts_obj=mcts_obj, action=action)
        action_type = 'nodes'
        log_prob = get_logprob(state, action, logits, mask_obj)
    return action_type, action, log_prob, logits, mask_obj

def get_logprob(state, action, logits, mask_obj):
    if state.done_graph:
        assert mask_obj is None
        log_prob = logits.log_prob(action.squeeze(-1))
    else:
        log_prob_stop = \
            action.aid_stop.type(torch.float)*F.logsigmoid(logits.logits_stop) + \
            torch.logical_not(action.aid_stop).type(torch.float)*F.logsigmoid(-logits.logits_stop)
        log_prob_start = \
            torch.logical_not(action.aid_stop).type(torch.float)*\
            F.log_softmax(logits.logits_aid_start, dim=1)[torch.arange(len(action.aid_start)), action.aid_start]
        log_prob_end = \
            torch.logical_not(action.aid_stop).type(torch.float)*\
            F.log_softmax(logits.logits_aid_end, dim=1)[torch.arange(len(action.aid_end)), action.aid_end]

        log_prob = log_prob_stop + log_prob_start + log_prob_end
        if state.graph_collated is not None:
            log_prob = torch.logical_not(
                state.graph_collated.mask.type(torch.float).to(log_prob.device)
            ).view(-1,1)*log_prob
        log_prob = log_prob.view(-1)
    return log_prob

def run_action_graph(state, policy, search_cfg, mcts_obj=None, action=None):
    if mcts_obj is not None:
        mcts_obj.init_action()
    embeddings = policy.get_embeddings(state)

    num_nodes = get_num_nodes(state)
    constraints_unsat, n_faces_unsat = get_faces_stats(state)

    # get first node
    logits_aid_start = policy.get_aid_start(state, **embeddings)
    if num_nodes is None:
        constraints_n_new_nodes_max = None # start of search
    else:
        constraints_n_new_nodes_max = \
            search_cfg.constraint_n_nodes_max - num_nodes # at most add MAX_NODE_COUNT
    mask_start = get_mask_initial(
        state, constraints_n_new_nodes_max=constraints_n_new_nodes_max,
        device=logits_aid_start.device)

    assert torch.all(torch.ge(torch.sum(mask_start, dim=-1), 0.0))
    if action is None:
        aid_start = sample_action_aid_node(logits_aid_start, mask_start, mcts_obj=mcts_obj)
    else:
        if mcts_obj is None:
            aid_start = action.aid_start
        else:
            aid_start = sample_action_aid_node(logits_aid_start, mask_start, mcts_obj=mcts_obj, action=action.aid_start)

    # get second node
    logits_aid_end = policy.get_aid_end(state, aid_start, **embeddings)
    constraints_sat_all_faces = constraints_n_new_nodes_max == 1 # ensure all constraints sat
    mask_face_token = get_mask_face_token(
        state, constraints_unsat, n_faces_unsat,
        constraints_sat_all_faces=constraints_sat_all_faces,
        device=logits_aid_end.device)

    if state.graph_collated is None:
        mask_end = mask_face_token
    else:
        mask_unselected_aids = get_mask_unselected_aids(state, aid_start, device=logits_aid_end.device)
        mask_end = torch.logical_and(mask_face_token, mask_unselected_aids)

    batch_index = torch.arange(len(aid_start))
    mask_end[batch_index, aid_start] = False
    if state.graph_collated is not None:
        nid_selected = \
            torch.logical_and(
                torch.eq(
                    state.graph_collated.graph_index.unsqueeze(-1),
                    batch_index.to(device=state.graph_collated.graph_index.device).unsqueeze(0)),
                torch.eq(
                    state.graph_collated.aid_index.unsqueeze(-1),
                    aid_start.unsqueeze(0)
                )).sum(dim=-1).nonzero().view(-1)
        nid_selected = \
            torch.eq(
                state.graph_collated.edge_index[0].unsqueeze(-1),
                nid_selected.unsqueeze(0)
            ).sum(dim=-1).nonzero().view(-1)
        nid_neighbors = state.graph_collated.edge_index[1][nid_selected]
        aid_neighbors = state.graph_collated.aid_index[nid_neighbors]
        bid_neighbors = state.graph_collated.graph_index[nid_neighbors]
        mask_end[bid_neighbors, aid_neighbors] = False

    assert torch.all(torch.ge(torch.sum(mask_end, dim=-1), 0.0))
    if action is None:
        aid_end = sample_action_aid_node(logits_aid_end, mask_end, mcts_obj=mcts_obj)
    else:
        if mcts_obj is None:
            aid_end = action.aid_end
        else:
            aid_end = sample_action_aid_node(logits_aid_end, mask_end, mcts_obj=mcts_obj, action=action.aid_end)

    # get stop token
    logits_stop = policy.get_stop_token(state, **embeddings)
    mask_stop_token, mask_terminal = \
        get_mask_stop_token(state, search_cfg, num_nodes, n_faces_unsat, device=logits_stop.device)
    if action is None:
        aid_stop = sample_action_aid_stop(logits_stop, mask_stop_token, mask_terminal, mcts_obj=mcts_obj)
    else:
        if mcts_obj is None:
            aid_stop = action.aid_stop
        else:
            aid_stop = sample_action_aid_stop(logits_stop, mask_stop_token, mask_terminal, mcts_obj=mcts_obj, action=action.aid_stop)

    action = ActionGraphObj(aid_stop, aid_start, aid_end)
    logits = LogitGraphObj(logits_stop, logits_aid_start, logits_aid_end)

    mask_stop = torch.logical_and(mask_stop_token, torch.logical_not(mask_terminal))
    mask_obj = mask_start, mask_end, mask_stop
    return action, logits, mask_obj

##############################################################
# Reward Interface
##############################################################

def get_reward_rho_binary_search(next_state_collated, model_surrogate, search_cfg, dataset_forward, dataset_inverse, graph_true,
                                 PSU_specific_train_reward=True, device=None):
    def evaluate_rho(rho_old, rho_new, r_li, reward_best, curve_best, curve_true, graph):
        r_new = rho3r(rho_old, rho_new, r_li)
        graph.update_radius(r_new, rho_new, is_zscore=dataset_forward.dataset.is_zscore_graph, g_stats=dataset_forward.dataset.g_stats)
        with torch.no_grad():
            model_surrogate_prediction = model_surrogate.inference(graph)
            reward, _ = get_reward_helper(
                curve_true, model_surrogate_prediction, search_cfg, dataset_forward, dataset_inverse,
                PSU_specific_train_reward=PSU_specific_train_reward)
            support = 0.0 * torch.tensor([get_support(g) for g in graph.g_li]).to(device)
            # print(f'@@@ ratio: {reward.mean()}, support: {support.mean()}')
            reward += support
        curve = get_curve_helper(model_surrogate_prediction)
        indices_to_update = torch.ge(reward.to(reward_best.device), reward_best)
        reward_best = torch.maximum(reward.to(reward_best.device), reward_best)
        curve_best = curve.update(torch.logical_not(indices_to_update), curve_best)
        return reward_best, curve_best, indices_to_update

    graph, r_li = \
        next_state_collated.graph_collated.get_fwd_model_graph(
            base_cell=DATASET_CFG['base_cell'],
            rm_redundant=DATASET_CFG['rm_redundant'],
            rho=next_state_collated.rho_collated[:,0],
            is_zscore=dataset_forward.dataset.is_zscore_graph
        )
    graph.to(device)

    curve_best = None
    reward_best = float('-inf') * torch.ones(len(graph.g_li), device=device)
    curve_true = next_state_collated.curve_collated

    rho_old = next_state_collated.rho_collated[:, 0]
    if search_cfg.n_iterations_rho_binary_search is None:
        # assert False
        rho_best = deepcopy(rho_old)
        for rho_new in next_state_collated.rho_collated.transpose(0, 1):
            reward_best, curve_best, indices_to_update = \
                evaluate_rho(rho_old, rho_new, r_li, reward_best, curve_best, graph)
            rho_best[indices_to_update] = rho_new[indices_to_update]
    else:
        rho_lower = search_cfg.constraint_rho_min * torch.ones_like(rho_old, dtype=torch.float, device=rho_old.device)
        rho_upper = search_cfg.constraint_rho_max * torch.ones_like(rho_old, dtype=torch.float, device=rho_old.device)

        reward_best, curve_best, _ = \
            evaluate_rho(rho_old, rho_lower, r_li, reward_best, curve_best, curve_true, graph)
        reward_best, curve_best, indices_upper_ge_lower = \
            evaluate_rho(rho_old, rho_upper, r_li, reward_best, curve_best, curve_true, graph)
        rho_best = deepcopy(rho_lower)
        rho_best[indices_upper_ge_lower] = rho_upper[indices_upper_ge_lower]

        for asdf in range(search_cfg.n_iterations_rho_binary_search):
            rho_mid = rho_lower + (asdf+1) / (search_cfg.n_iterations_rho_binary_search+1) * (rho_upper - rho_lower)
            reward_best, curve_best, indices_to_update = \
                evaluate_rho(rho_old, rho_mid, r_li, reward_best, curve_best, curve_true, graph)
            rho_best[indices_to_update] = rho_mid[indices_to_update]

    r_best = rho3r(rho_old, rho_best, r_li)
    graph.update_radius(r_best, rho_best, is_zscore=dataset_forward.dataset.is_zscore_graph, g_stats=dataset_forward.dataset.g_stats)
    for g, rho, r in zip(graph.g_li, rho_best, r_best):
        g.graph['rho'] = rho.item()
        for eid in g.edges():
            g.edges[eid]['radius'] = r
    *_, cn = model_surrogate.inference(graph, return_cn=True)
    for i, cn_elt in enumerate(cn):
        C, n = cn_elt
        graph.g_li[i].graph['C'] = C.item()
        graph.g_li[i].graph['n'] = n.item()
    return reward_best, graph, curve_best

def get_reward(next_state_collated, model_surrogate, search_cfg, dataset, graph_true, device=None):
    assert False
    graph, r_li = \
        next_state_collated.graph_collated.get_fwd_model_graph(
            base_cell=DATASET_CFG['base_cell'],
            rm_redundant=DATASET_CFG['rm_redundant'],
            rho=next_state_collated.rho_collated[:,0],
            is_zscore=dataset.dataset.is_zscore_graph
        )

    graph.to(device)

    curve_best = None
    reward_best = float('-inf') * torch.ones(len(graph.g_li), device=device)
    curve_true = next_state_collated.curve_collated
    for rho_new in next_state_collated.rho_collated.transpose(0,1):
        r_new = rho3r(next_state_collated.rho_collated[:,0], rho_new, r_li)
        graph.update_radius(r_new, rho_new, is_zscore=dataset.dataset.is_zscore_graph, g_stats=dataset.dataset.g_stats)

        # ########################## TOY CASE ##########################
        # from src.dataset_feats_edge import EDGE_FEAT_CFG
        # c_magnitude = scatter_add(graph.feats_edge[:,1], graph.graph_edge_index).unsqueeze(-1)
        # c_magnitude_mean, c_magnitude_std, _, _ = dataset.dataset.c_stats
        # c_magnitude = (c_magnitude - c_magnitude_mean.to(c_magnitude.device)) / (c_magnitude_std.to(c_magnitude.device)+1e-10)
        # model_surrogate_prediction = curve_true.c_shape, c_magnitude, 0.0*curve_true.c_shape, 0.0*c_magnitude
        # ##############################################################
        with torch.no_grad():
            model_surrogate_prediction = model_surrogate.inference(graph)
            reward, _ = get_reward_helper(curve_true, model_surrogate_prediction, search_cfg, dataset)
            support = 0.0 * torch.tensor([get_support(g) for g in graph.g_li]).to(device)
            print(f'@@@ ratio: {reward.mean()}, support: {support.mean()}')
            reward += support
        curve = get_curve_helper(model_surrogate_prediction)

        indices_to_update = torch.ge(reward.to(reward_best.device), reward_best)
        reward_best = torch.maximum(reward.to(reward_best.device), reward_best)
        curve_best = curve.update(torch.logical_not(indices_to_update), curve_best)
    return reward_best, graph, curve_best

def get_reward_helper(curve_true, model_surrogate_prediction, search_cfg, dataset,
                      silent=False, skip_processing=False, PSU_specific_train_reward=True): # v2
    c_shape, c_magnitude, c_shape_uncertainty, c_magnitude_uncertainty = model_surrogate_prediction
    c_pred_mean, (c_pred_UB, c_pred_LB) = \
        dataset.dataset.unnormalize_curve(
            c_magnitude, c_shape,
            c_magnitude_u=c_magnitude_uncertainty,
            c_shape_u=c_shape_uncertainty)
    c_pred = c_pred_mean
    c_true = \
        dataset.dataset.unnormalize_curve(
            curve_true.c_magnitude, curve_true.c_shape)[0]
    r_uncertainty = \
        torch.mean((c_pred_UB-c_pred_LB).view(c_pred.shape[0], -1), dim=-1) / \
        (torch.max(c_pred.view(c_pred.shape[0], -1), dim=-1)[0]+1e-12)
    if TASK == 'stress_strain':
        union = torch.where(torch.ge(c_true,c_pred), c_true, c_pred).squeeze(2)
        intersection = torch.where(torch.ge(c_true,c_pred), c_pred, c_true).squeeze(2)
        r_jaccard = torch.sum(intersection, dim=-1) / torch.sum(union, dim=-1)
        r_main = r_jaccard
    elif TASK == 'transmission':
        if dataset.dataset.digitize_cfg is None:
            # c_pred += 40.0
            # c_true += 40.0
            # numer = torch.abs(c_pred - c_true).squeeze(-1)
            # denom = (c_true.squeeze(-1).max(dim=-1)[0] - c_true.squeeze(-1).min(dim=-1)[0]).unsqueeze(-1)
            # r_main = (numer/denom).mean(-1)
            # c_pred -= 40.0
            # c_true -= 40.0
            c_pred_ = torch.clamp(c_pred+40.0, min=0.0)
            c_true_ = torch.clamp(c_true+40.0, min=0.0)
            union = torch.where(torch.ge(c_true_,c_pred_), c_true_, c_pred_).squeeze(2)
            intersection = torch.where(torch.ge(c_true_,c_pred_), c_pred_, c_true_).squeeze(2)
            r_jaccard = torch.sum(intersection, dim=-1) / torch.sum(union, dim=-1)
            r_main = r_jaccard
        else:
            if not skip_processing:
                c_pred = dataset.dataset.digitize_curve_torch(c_pred).unsqueeze(-1)

            bin, = [float(x) for x in dataset.dataset.digitize_cfg['bins']]

            if PSU_specific_train_reward:
                r_main = \
                    (torch.logical_xor(
                        c_true.to(torch.bool),
                        torch.gt(c_pred, bin)).float() * -torch.abs(c_pred-bin)
                    ).squeeze(dim=-1).mean(dim=-1)
            else:
                r_main = (c_pred == c_true).float().squeeze(dim=-1).mean(dim=-1)

    if not silent:
        print(f'rewards: \
         jaccard={to_item(r_main.mean()):.3f}{PLUS_MINUS}{to_item(r_main.std()):.3f}')
    reward = \
        search_cfg.jaccard_coeff * r_main + \
        search_cfg.uncertainty_coeff * r_uncertainty
    other = {
        'jaccard': r_main.min(),
        'uncertainty': r_uncertainty.min()
    }
    return reward, other

##############################################################
# Environment Interface
##############################################################

def run_environment_collated(
        action_type, action_collated, cur_state_collated, model_surrogate, graph_true,
        g_stats, node_feats_index, edge_feats_index, search_cfg, dataset_forward, dataset_inverse,
        PSU_specific_train_reward=True, device=None):
    next_state_collated = deepcopy(cur_state_collated)
    if action_type == 'rho':
        next_state_collated.rho_collated = action_collated
        next_state_collated.done_rho = True
        if model_surrogate is None:
            reward, graph, curve = 0.0, None, None
        else:
            #get_reward_rho_binary_search
            reward, graph, curve = \
                get_reward_rho_binary_search(
                    next_state_collated, model_surrogate, search_cfg, dataset_forward, dataset_inverse, graph_true,
                    PSU_specific_train_reward=PSU_specific_train_reward, device=device)
    elif action_type == 'nodes':
        if next_state_collated.graph_collated is None:
            next_state_collated.graph_collated = \
                GraphObjCollatedv2.from_first_nodes(
                    action_collated.aid_start, g_stats, node_feats_index, edge_feats_index,
                    dataset_forward.dataset.edge_feat_cfg)
            next_state_collated.graph_collated.add_new_edges(
                action_collated.aid_start, action_collated.aid_end)
        else:
            mask_cur = next_state_collated.graph_collated.mask
            next_state_collated.graph_collated.mask = \
                torch.logical_or(mask_cur, action_collated.aid_stop.view(-1).to(mask_cur.device))
            if False in next_state_collated.graph_collated.mask:
                next_state_collated.graph_collated.add_new_edges(
                    action_collated.aid_start, action_collated.aid_end)
            else:
                next_state_collated.done_graph = True
        reward = 0.0
        graph = None
        curve = None
    else:
        assert False
    return next_state_collated, reward, graph, curve

def sample_entropy(action_type, logits, action):
    if action_type == 'nodes':
        entropy_start = torch.distributions.Categorical(logits=logits.logits_aid_start.squeeze(-1)).entropy().mean()
        entropy_end = torch.distributions.Categorical(logits=logits.logits_aid_end.squeeze(-1)).entropy().mean()
        entropy = entropy_start + entropy_end
    elif action_type == 'rho':
        entropy = logits.entropy().mean()
    else:
        assert False
    return entropy

##############################################################
# Utility Functions
##############################################################

def get_curve_helper(model_surrogate_prediction):
    c_shape, c_magnitude, c_shape_uncertainty, c_magnitude_uncertainty = model_surrogate_prediction
    curve = CObj(
        c_magnitude=c_magnitude,
        c_shape=c_shape,
        c_magnitude_std=c_magnitude_uncertainty,
        c_shape_std=c_shape_uncertainty)
    return curve

def get_jaccard(c_true,c_pred):
    union = torch.where(torch.ge(c_true,c_pred), c_true, c_pred).squeeze(2)
    intersection = torch.where(torch.ge(c_true,c_pred), c_pred, c_true).squeeze(2)
    r_jaccard = torch.sum(intersection, dim=-1) / torch.sum(union, dim=-1)
    return r_jaccard

def get_num_nodes(state):
    graph = state.graph_collated
    if graph is not None:
        batch_index, num_nodes_scattered = \
            torch.unique(graph.graph_index[graph.graph_base_index], return_counts=True) # b x 1, b x 1
        num_nodes = torch.zeros_like(batch_index)
        num_nodes[batch_index] = num_nodes_scattered
    else:
        num_nodes = None
    return num_nodes

def get_faces_stats(state):
    graph = state.graph_collated
    if graph is not None:
        constraints_cur_all = \
            ACTION_CONSTRAINTS[graph.aid_index].to(graph.graph_index.device)  # n_nodes x n_faces
        constraints_sat = \
            scatter_add(
                constraints_cur_all.type(torch.float), graph.graph_index,
                dim=0, dim_size=len(state.curve_collated))  # b x n_faces
        constraints_unsat = torch.eq(constraints_sat, 0)  # b x n_faces
        n_faces_unsat = constraints_unsat.sum(dim=-1)

        constraints_nfaces = \
            torch.sum(torch.clamp(scatter_add(
                ACTION_CONSTRAINTS[graph.aid_index].type(torch.float).to(graph.graph_index.device),
                graph.graph_index, dim=0, dim_size=constraints_unsat.shape[0]), min=0.0, max=1.0), dim=-1)
        # constraints_sat = torch.eq(constraints_nfaces, 4)
        assert torch.all(constraints_nfaces == 4 - n_faces_unsat)
    else:
        constraints_unsat = None
        n_faces_unsat = None
    return constraints_unsat, n_faces_unsat

##############################################################
# Defects and Model Extensions
##############################################################

def get_support(g):
    supported_nodes = mark_supported_nodes(g, z_min=-1)
    propagate_front(g, supported_nodes)
    ratio_i = len(supported_nodes) / len(g.nodes())
    return ratio_i

def mark_supported_nodes(graph, z_min=-1, eps=1e-06):
    supported_nodes = set()

    for node in graph.nodes():
        if abs(graph.nodes[node]['coord'][2] - z_min) < eps:
            supported_nodes.add(node)
            continue
        for neighbor in graph.neighbors(node):
            if graph.nodes[node]['coord'][2] > graph.nodes[neighbor]['coord'][2]:
                supported_nodes.add(node)
                break
    return supported_nodes

def propagate_front(graph, supported_nodes, eps=1e-06):
    visited = set()
    queue = list(supported_nodes)

    while queue:
        current_node = queue.pop(0)
        visited.add(current_node)

        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited and abs(
                    graph.nodes[current_node]['coord'][2] - graph.nodes[neighbor]['coord'][2]) < eps:
                queue.append(neighbor)
                supported_nodes.add(neighbor)
    return supported_nodes