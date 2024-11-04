import os
import csv
import pickle
import torch
import math
import random
import itertools
import networkx as nx
import numpy as np
from collections import defaultdict
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from itertools import combinations
from scipy import spatial
from copy import deepcopy
from tqdm import tqdm
from skimage.measure import block_reduce

from src.utils import compute_lengths, rho2r, DoubleDict, plot_3d
from src.dataset_curve import get_curve, unnormalize_curve, DEFAULT_MAGNITUDE
from src.dataset_feats_edge import EDGE_FEAT_CFG, get_edge_li, get_edge_index, get_edge_feats
from src.dataset_feats_node import NODE_FEAT_CFG, get_node_feats
from src.generative_graph.tesellate import tesselate, tesselate_torch, rm_redundant_nodes, rm_double_edges
from src.dataset_preprocessing_collated import DATASET_CFG
from src.config import args, ETH_FULL_C_VECTOR
from src.gen_unseen_curve import gen_unseen_seq

EDGE_DISCR = 5
USE_BINNING = False

def get_action_space_and_constraints():
    tetrahedron_nodes = [
        torch.tensor([0,0,0]),
        torch.tensor([1,1,1]),
        torch.tensor([1,1,0]),
        torch.tensor([1,0,0]),
    ]
    tetrahedron_face_constraints = [
        torch.tensor([True, True, True, False]),
        torch.tensor([True, False, True, True]),
        torch.tensor([True, True, False, True]),
        torch.tensor([False, True, True, True]),
    ]
    tetrahedron = nx.Graph()
    tetrahedron.add_nodes_from(list(range(len(tetrahedron_face_constraints))))
    assert len(tetrahedron_nodes) == len(tetrahedron_face_constraints)

    additional_nodes = []
    additional_face_constraints = []
    for idx1, idx2 in itertools.combinations(list(range(len(tetrahedron_nodes))), 2):
        idx1, idx2 = sorted([idx1, idx2])
        u = tetrahedron_nodes[idx1]
        v = tetrahedron_nodes[idx2]
        u_face_constraints = tetrahedron_face_constraints[idx1]
        v_face_constraints = tetrahedron_face_constraints[idx2]

        prev_idx, next_idx = idx1, tetrahedron.number_of_nodes()
        for t in np.linspace(0,1,num = EDGE_DISCR)[1:-1]:
            w = (1-t)*u + t*v
            w_face_constraints = \
                torch.logical_and(u_face_constraints, v_face_constraints)
            additional_nodes.append(w)
            additional_face_constraints.append(w_face_constraints)

            assert next_idx == len(additional_nodes) + len(tetrahedron_nodes) - 1
            tetrahedron.add_edge(prev_idx, next_idx) # connect the chain
            prev_idx, next_idx = tetrahedron.number_of_nodes()-1, tetrahedron.number_of_nodes()
        tetrahedron.add_edge(prev_idx, idx2) # finish the chain
    assert len(additional_nodes) == len(additional_face_constraints)

    action_space =\
        torch.stack(
            tetrahedron_nodes + additional_nodes, dim=0)
    action_constraints = \
        torch.stack(
            tetrahedron_face_constraints + additional_face_constraints, dim=0)
    return action_space, action_constraints, tetrahedron

ACTION_SPACE, ACTION_CONSTRAINTS, TETRAHEDRON = get_action_space_and_constraints()

AID2FEATS = deepcopy(ACTION_SPACE).type(torch.float)

def get_aid_bw_li(aid_start_li, aid_end_li, aid_sel_li):
    aid_bw_li = []
    aid_mask_li = []
    for aid_start, aid_end, aid_sel in zip(aid_start_li, aid_end_li, aid_sel_li):
        overlapped_face = \
            torch.logical_and(ACTION_CONSTRAINTS[aid_start], ACTION_CONSTRAINTS[aid_end])
        if torch.sum(overlapped_face) == 2: # lies on an edge
            mask_edge_signature = torch.sum(overlapped_face.unsqueeze(0)==ACTION_CONSTRAINTS, dim=-1) == 4
            bbox_max = torch.maximum(ACTION_SPACE[aid_start], ACTION_SPACE[aid_end])
            bbox_min = torch.minimum(ACTION_SPACE[aid_start], ACTION_SPACE[aid_end])
            mask_within_bbox = \
                torch.logical_and(
                    torch.sum(ACTION_SPACE <= bbox_max, dim=-1) == 3,
                    torch.sum(ACTION_SPACE >= bbox_min, dim=-1) == 3)
            aid_bw = \
                torch.where(torch.logical_and(
                    mask_within_bbox, mask_edge_signature))[0].detach().cpu().tolist()

            if aid_start in aid_bw:
                aid_bw.remove(aid_start)
            if aid_end in aid_bw:
                aid_bw.remove(aid_end)

            if len(aid_bw) > 0:
                if torch.norm(ACTION_SPACE[aid_start] - ACTION_SPACE[min(aid_bw)]) <=\
                        torch.norm(ACTION_SPACE[aid_start] - ACTION_SPACE[max(aid_bw)]):
                    aid_bw = sorted(aid_bw)
                else:
                    aid_bw = sorted(aid_bw, reverse=True)

            aid_mask = [(aid not in aid_sel) for aid in aid_bw]

        else:
            aid_bw = []
            aid_mask = []
        aid_bw_li.append(aid_bw)
        aid_mask_li.append(aid_mask)
    return aid_bw_li, aid_mask_li

class DataLoaderFactory:
    def __init__(self, node_feat_cfg, edge_feat_cfg, train_split, valid_split, test_split,
                 is_zscore_graph, is_zscore_curve_magni, is_zscore_curve_shape, batch_size,
                 num_workers, augment_graph=False, augment_curve=False, root_graph=None, root_curve=None,
                 root_mapping=None, root_graph_pre=None, root_prop_pre=None, root_mapping_pre=None,
                 curve_norm_cfg=None, digitize_cfg=None, use_cosine_node=0, use_cosine_edge=0, apply_patch=False):
        self.root_graph = root_graph
        self.root_curve = root_curve
        self.root_mapping = root_mapping
        self.root_graph_pre = root_graph_pre
        self.root_prop_pre = root_prop_pre
        self.root_mapping_pre = root_mapping_pre
        self.node_feat_cfg = node_feat_cfg
        self.edge_feat_cfg = edge_feat_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.valid_split = valid_split
        self.test_split = test_split
        self.is_zscore_graph = is_zscore_graph
        self.is_zscore_curve_magni = is_zscore_curve_magni
        self.is_zscore_curve_shape = is_zscore_curve_shape
        self.curve_norm_cfg = {} if curve_norm_cfg is None else curve_norm_cfg
        self.use_cosine_node = use_cosine_node
        self.use_cosine_edge = use_cosine_edge
        self.augment_graph = augment_graph
        self.augment_curve = augment_curve
        self.digitize_cfg = digitize_cfg

        split = train_split #'train'
        binning_pn = os.path.join(self.root_curve, split, 'binning', f'binning_dict.pkl')
        if os.path.exists(binning_pn) and USE_BINNING:
            print('Loading binned weights for curves')
            with open(binning_pn, 'rb') as fp:
                binning_dict = pickle.load(fp)
        else:
            binning_dict = None
        self.binning_dict = binning_dict
        self.train_dataset = \
            GraphCurveDataset(
                graph_pn=os.path.join(self.root_graph, split, 'graphs'),
                curve_pn=os.path.join(self.root_curve, split, 'curves'),
                mapping_pn=os.path.join(self.root_mapping, split),
                split=split,
                curve_norm_cfg=self.curve_norm_cfg,
                node_feat_cfg=self.node_feat_cfg,
                edge_feat_cfg=self.edge_feat_cfg,
                digitize_cfg=self.digitize_cfg,
                is_zscore_graph=self.is_zscore_graph,
                is_zscore_curve_magni=self.is_zscore_curve_magni,
                is_zscore_curve_shape=self.is_zscore_curve_shape,
                use_cosine_node=self.use_cosine_node,
                use_cosine_edge=self.use_cosine_edge,
                binning_dict=self.binning_dict,
                apply_patch=apply_patch
            )
        # unseen_se = gen_unseen_seq(
        #     '/'.join(self.root_graph.split('/')[:-1] + ['Fine_tuning_data']),
        #     self.digitize_cfg['n_freq'], shuffle_on=False, dist_limit=0.25, un_curves=1_000)
        # self.unseen_se_train = unseen_se[:int(0.9*len(unseen_se))]
        # self.unseen_se_valid = unseen_se[int(0.9*len(unseen_se)):int(0.95*len(unseen_se))]
        # self.unseen_se_test = unseen_se[int(0.95*len(unseen_se)):]

    def get_train_dataset_contrastive(self):
        return self._get_dataset_contrastive(
            self.train_split, shuffle=True, binning_dict=self.binning_dict,
            augment_graph=self.augment_graph, augment_curve=self.augment_curve)

    def get_train_dataset(self):
        return self._get_dataset(
            self.train_split, shuffle=True, binning_dict=self.binning_dict,
            augment_graph=self.augment_graph, augment_curve=self.augment_curve)
        # return self._get_dataset_custom(self.unseen_se_train, self.test_split, shuffle=True, binning_dict=self.binning_dict)

    def get_valid_dataset(self):
        return self._get_dataset(self.valid_split, shuffle=True, binning_dict=None)
        # return self._get_dataset_custom(self.unseen_se_valid, self.test_split, shuffle=True, binning_dict=None)

    def get_test_dataset(self):
        return self._get_dataset(self.test_split, shuffle=False, binning_dict=None)
        # return self._get_dataset_custom(self.unseen_se_test, self.test_split, shuffle=False, binning_dict=None)

    def _get_dataset_custom(self, unseen_se, split, shuffle=True, augment_graph=False, augment_curve=False, binning_dict=None):
        dataset = \
            CurveOverideDataset(
                unseen_se,
                graph_pn=os.path.join(self.root_graph, split, 'graphs'),
                curve_pn=os.path.join(self.root_curve, split, 'curves'),
                mapping_pn=os.path.join(self.root_mapping, split),
                split=split,
                node_feat_cfg=self.node_feat_cfg,
                edge_feat_cfg=self.edge_feat_cfg,
                curve_norm_cfg=self.curve_norm_cfg,
                digitize_cfg=self.digitize_cfg,
                g_stats=self.train_dataset.g_stats,
                c_stats=self.train_dataset.c_stats,
                is_zscore_curve_magni=self.is_zscore_curve_magni,
                is_zscore_curve_shape=self.is_zscore_curve_shape,
                is_zscore_graph=self.is_zscore_graph,
                use_cosine_node=self.use_cosine_node,
                use_cosine_edge=self.use_cosine_edge,
                binning_dict=binning_dict,
                augment_graph=augment_graph,
                augment_curve=augment_curve
            )
        return DataLoader(
            dataset,
            batch_size=min(128, self.batch_size) if split == 'test' else self.batch_size,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
        )

    def _get_dataset(self, split, shuffle=True, augment_graph=False, augment_curve=False, binning_dict=None):
        if split == 'train':
            dataset = self.train_dataset
        else:
            dataset = \
                GraphCurveDataset(
                    graph_pn=os.path.join(self.root_graph, split, 'graphs'),
                    curve_pn=os.path.join(self.root_curve, split, 'curves'),
                    mapping_pn=os.path.join(self.root_mapping, split),
                    split=split,
                    node_feat_cfg=self.node_feat_cfg,
                    edge_feat_cfg=self.edge_feat_cfg,
                    curve_norm_cfg=self.curve_norm_cfg,
                    digitize_cfg=self.digitize_cfg,
                    g_stats=self.train_dataset.g_stats,
                    c_stats=self.train_dataset.c_stats,
                    is_zscore_curve_magni=self.is_zscore_curve_magni,
                    is_zscore_curve_shape=self.is_zscore_curve_shape,
                    is_zscore_graph=self.is_zscore_graph,
                    use_cosine_node=self.use_cosine_node,
                    use_cosine_edge=self.use_cosine_edge,
                    binning_dict=binning_dict,
                    augment_graph=augment_graph,
                    augment_curve=augment_curve
                )
        return DataLoader(
            dataset,
            batch_size=min(128, self.batch_size) if split == 'test' else self.batch_size,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
        )

    def _get_dataset_contrastive(self, split, shuffle=True, augment_graph=False, augment_curve=False, binning_dict=None):
        dataset = \
            GraphCurveDatasetContrastive(
                graph_pn=os.path.join(self.root_graph, split, 'graphs'),
                curve_pn=os.path.join(self.root_curve, split, 'curves'),
                mapping_pn=os.path.join(self.root_mapping, split),
                split=split,
                curve_norm_cfg=self.curve_norm_cfg,
                node_feat_cfg=self.node_feat_cfg,
                edge_feat_cfg=self.edge_feat_cfg,
                digitize_cfg=self.digitize_cfg,
                is_zscore_graph=self.is_zscore_graph,
                is_zscore_curve_magni=self.is_zscore_curve_magni,
                is_zscore_curve_shape=self.is_zscore_curve_shape,
                use_cosine_node=self.use_cosine_node,
                use_cosine_edge=self.use_cosine_edge,
                binning_dict=self.binning_dict
            )
        return DataLoader(
            dataset,
            batch_size=min(128, self.batch_size) if split == 'test' else self.batch_size,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
        )

def bins2weight(data, cid, num_classes):
    total_samples = len(data)
    num_samples_in_class_i = data[cid]['count']
    num_classes = num_classes
    weight = total_samples / (num_samples_in_class_i * num_classes)
    return weight

from scipy.interpolate import CubicSpline      # for warping
def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    out = np.array([CubicSpline(xx[:,i], yy[:,i])(x_range) for i in range(xx.shape[1])]).transpose()
    return out

def timewarp(x, sigma):
    x_range = np.arange(x.shape[0])
    tt = GenerateRandomCurves(x, sigma) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    out = []
    for i in range(x.shape[1]):
        t_scale = (x.shape[0] - 1) / tt_cum[-1, i]
        tt_cum[:, i] = tt_cum[:, i] * t_scale
        out.append(np.interp(x_range, tt_cum[:, i], x[:, i]))
    return np.stack(out, axis=1)

def magnitudewarp(x, sigma):
    return x * GenerateRandomCurves(x, sigma)


class GraphCurveDataset(Dataset):
    def __init__(self, graph_pn, curve_pn, mapping_pn, split, is_zscore_graph,
                 is_zscore_curve_magni, is_zscore_curve_shape,
                 node_feat_cfg, edge_feat_cfg, curve_norm_cfg, digitize_cfg=None,
                 augment_graph=False, augment_curve=False, g_stats=None, c_stats=None,
                 use_cosine_node=0, use_cosine_edge=0, binning_dict=None, apply_patch=False):
        print('initializing mappings...')
        self.apply_patch = apply_patch
        self.mapping, self.gid2graph_pn, self.cid2curve_pn = \
            self._init_mapping(mapping_pn, graph_pn, curve_pn)
        self.curve_norm_cfg = curve_norm_cfg
        self.is_labeled = mapping_pn is not None
        self.split = split
        self.use_cosine_node = use_cosine_node
        self.use_cosine_edge = use_cosine_edge
        self.node_feat_cfg = sorted(node_feat_cfg, key=lambda x: NODE_FEAT_CFG.index(x))
        self.edge_feat_cfg = sorted(edge_feat_cfg, key=lambda x: EDGE_FEAT_CFG.index(x))
        self.node_feats_index = [NODE_FEAT_CFG.index(x) for x in node_feat_cfg]
        self.edge_feats_index = [EDGE_FEAT_CFG.index(x) for x in edge_feat_cfg]
        self.is_zscore_graph = is_zscore_graph
        self.is_zscore_curve_magni = is_zscore_curve_magni
        self.is_zscore_curve_shape = is_zscore_curve_shape
        self.binning_dict = binning_dict
        self.augment_curve = augment_curve
        self.augment_graph = augment_graph
        self.digitize_cfg = digitize_cfg

        print('getting statistics...')
        self.g_stats = self._get_g_stats(max_samples=20_000) if g_stats is None else g_stats
        self.c_stats = self._get_c_stats(max_samples=20_000) if c_stats is None else c_stats

        graph_sample = self._get_graph(
            list(self.gid2graph_pn.values())[0])
        curve_sample = self._get_curve(
            list(self.cid2curve_pn.values())[0],
            is_zscore_curve_magni=self.is_zscore_curve_magni,
            is_zscore_curve_shape=self.is_zscore_curve_shape)
        self.dim_input_nodes = graph_sample.feats_node.shape[-1]
        self.dim_input_edges = graph_sample.feats_edge.shape[-1]
        self.dim_curve = curve_sample.c_shape.shape[-1]

    def _init_mapping(self, mapping_pn, graph_pn, curve_pn):
        '''
        Args:
            mapping_pn: root directory to tsv files
                tsv file is tab separated gid, cid pairs
            split: train/valid/test split

        Returns:
            MappingObj object
        '''
        mappings = MappingObj(os.path.join(mapping_pn, 'mapping.tsv'))
        gid2graph_pn, cid2curve_pn = {}, {}
        for gid, cid in tqdm(mappings.gid_cid_li):
            graph_fn = os.path.join(graph_pn, f'{gid}.gpkl')
            curve_fn = os.path.join(curve_pn, f'{cid}.pkl')
            # with open(graph_fn, 'rb') as f:
            #     g = pickle.load(f)
            #     assert gid == g.graph['gid']
            # with open(curve_fn, 'rb') as fp:
            #     c = pickle.load(fp)
            #     assert cid == c['cid']
            gid2graph_pn[gid] = graph_fn
            cid2curve_pn[cid] = curve_fn
        return mappings, gid2graph_pn, cid2curve_pn

    def __len__(self):
        return len(self.mapping.gid_cid_li)

    def __getitem__(self, idx):
        gid, cid = self.mapping.gid_cid_li[idx]
        graph_pn = self.gid2graph_pn[gid]
        curve_pn = self.cid2curve_pn[cid]
        graph = self._get_graph(graph_pn, zscore=self.is_zscore_graph)
        # curve = self._get_curve_dummy(graph_pn, curve_pn, zscore=self.is_zscore_curve)
        curve = self._get_curve(
            curve_pn,
            is_zscore_curve_magni=self.is_zscore_curve_magni,
            is_zscore_curve_shape=self.is_zscore_curve_shape,
        )
        return graph, curve

    def collate_fn(self, samples):
        graph_li, curve_li = zip(*samples)
        graph_collated = GraphObjCollated.collate(graph_li)
        curve_collated = CurveObjCollated.collate(curve_li)
        return graph_collated, curve_collated

    def _get_g_stats(self, max_samples, seed=123):#55):
        random.seed(seed)
        graph_pn_li_sample = \
            [
                self.gid2graph_pn[gid] for gid in \
                random.sample(
                    sorted(self.gid2graph_pn.keys()),
                    k=min(max_samples, len(self.gid2graph_pn)))
            ]

        node_feats_all, edge_feats_all, rho_feats_all = [], [], []
        for graph_pn in tqdm(graph_pn_li_sample):
            graph = self._get_graph(graph_pn, zscore=False)
            node_feats_all.append(graph.feats_node)
            edge_feats_all.append(graph.feats_edge)
            rho_feats_all.append(graph.g.graph['rho'])
        node_feats_mean = \
            torch.mean(torch.cat(node_feats_all, dim=0), dim=0).view(1, -1)  # 1 x dim_node
        node_feats_std = \
            torch.std(torch.cat(node_feats_all, dim=0), dim=0).view(1, -1)  # 1 x dim_node
        edge_feats_mean = \
            torch.mean(torch.cat(edge_feats_all, dim=0), dim=0).view(1, -1)  # 1 x dim_node
        edge_feats_std = \
            torch.std(torch.cat(edge_feats_all, dim=0), dim=0).view(1, -1)  # 1 x dim_node
        rho_feats_mean = \
            torch.mean(torch.tensor(rho_feats_all))
        rho_feats_std = \
            torch.std(torch.tensor(rho_feats_all))
        g_stats = \
            (node_feats_mean, node_feats_std,
             edge_feats_mean, edge_feats_std,
             rho_feats_mean, rho_feats_std)
        return g_stats

    def _get_c_stats(self, max_samples, seed=123):#56):
        random.seed(seed)
        curve_pn_li_sample = \
            [
                self.cid2curve_pn[cid] for cid in \
                random.sample(
                    sorted(self.cid2curve_pn.keys()),
                    k=min(max_samples, len(self.cid2curve_pn)))
            ]

        c_magnitude_all, c_shape_all = [], []
        for curve_pn in tqdm(curve_pn_li_sample):
            curve = self._get_curve(curve_pn,  is_zscore_curve_magni=False, is_zscore_curve_shape=False)
            c_magnitude_all.append(curve.c_magnitude)
            c_shape_all.append(curve.c_shape)

        if type(self.is_zscore_curve_magni) == str and self.is_zscore_curve_magni == 'minmax':
            c_magnitude_mean = torch.mean(torch.stack(c_magnitude_all, dim=0), dim=0)  # 1
            c_magnitude_std = torch.std(torch.stack(c_magnitude_all, dim=0), dim=0)  # 1
        else:
            assert self.is_zscore_curve_magni is None or type(self.is_zscore_curve_magni) == bool
            c_magnitude_mean = torch.mean(torch.stack(c_magnitude_all, dim=0), dim=0)  # 1
            c_magnitude_std = torch.std(torch.stack(c_magnitude_all, dim=0), dim=0)  # 1

        if type(self.is_zscore_curve_shape) == str and self.is_zscore_curve_shape == 'minmax':
            c_shape_mean = torch.min(torch.stack(c_shape_all))  # 1 x resolution x 1
            c_shape_std = torch.max((torch.stack(c_shape_all) - c_shape_mean).view(-1))  # 1
        else:
            assert self.is_zscore_curve_shape is None or type(self.is_zscore_curve_shape) == bool
            from src.config import ETH_FULL_C_VECTOR
            if ETH_FULL_C_VECTOR:
                c_shape_mean = torch.mean(torch.stack(c_shape_all, dim=0), dim=0).unsqueeze(0)  # 1 x resolution x 1
            else:
                c_shape_mean = torch.mean(torch.stack(c_shape_all, dim=0), dim=0)  # 1 x resolution x 1
            c_shape_std = torch.std((torch.stack(c_shape_all, dim=0) - c_shape_mean).view(-1))  # 1

        c_stats = c_magnitude_mean, c_magnitude_std, c_shape_mean, c_shape_std
        return c_stats

    def _get_graph(self, graph_pn, zscore=True, eps=1e-10):
        with open(graph_pn, 'rb') as f:
            g = pickle.load(f)

        # get g_poly
        g_poly = None
        if os.path.exists(os.path.join(graph_pn.replace('.gpkl', '_polyhedron.gpkl'))):
            with open(os.path.join(graph_pn.replace('.gpkl', '_polyhedron.gpkl')), 'rb') as f:
                g_poly = pickle.load(f)
        g.graph['polyhedron'] = g_poly

        # get tensors
        edge_index = torch.LongTensor(g.graph.pop('edge_index'))

        node_feats = torch.FloatTensor(g.graph.pop('node_feats')[:,self.node_feats_index])
        if self.use_cosine_node > 0:
            node_feats_more = []
            for k in range(self.use_cosine_node):
                node_feats_more.append(torch.cos(2*np.pi*(k+1)*node_feats))
                node_feats_more.append(-torch.cos(2*np.pi*(k+1)*node_feats))
            node_feats = torch.cat(node_feats_more, dim=-1)

        edge_feats = torch.FloatTensor(g.graph.pop('edge_feats')[:,self.edge_feats_index])
        if self.use_cosine_edge > 0:
            edge_feats_more = []
            for k in range(self.use_cosine_edge):
               edge_feats_more.append(torch.cos(np.pi/4*(k+1)*edge_feats))
               edge_feats_more.append(torch.sin(np.pi/4*(k+1)*edge_feats))
            edge_feats_more = torch.cat([edge_feats] + edge_feats_more, dim=-1)

        assert edge_index.shape[1] == edge_feats.shape[0] == 2*g.number_of_edges()
        assert node_feats.shape[0] == g.number_of_nodes()


        # edge_li = get_edge_li(g)
        # edge_index = get_edge_index(edge_li)
        # node_feats = get_node_feats(g, self.node_feat_cfg, ldp_args={'edge_index': edge_index}, manual_args={})
        # edge_feats = get_edge_feats(g, self.edge_feat_cfg, edge_li=edge_li)
        rho_feats = g.graph['rho']

        if zscore:
            (node_feats_mean, node_feats_std,
             edge_feats_mean, edge_feats_std,
             rho_feats_mean, rho_feats_std) = self.g_stats
            node_feats_zscore = (node_feats - node_feats_mean) / (node_feats_std + eps)
            edge_feats_zscore = (edge_feats - edge_feats_mean) / (edge_feats_std + eps)
            # rho_feats_zscore = (rho_feats - rho_feats_mean) / (rho_feats_std + eps)
            rho_feats_zscore = rho_feats
            if self.augment_graph:
                scale = 0.2
                node_feats_delta = torch.clip(scale*torch.randn_like(node_feats_zscore), min=-scale, max=scale)
                edge_feats_delta = torch.clip(scale*torch.randn_like(edge_feats_zscore), min=-scale, max=scale)
                node_feats_zscore = node_feats_zscore + node_feats_delta
                edge_feats_zscore = edge_feats_zscore + edge_feats_delta
        else:
            rho_feats_zscore = rho_feats
            node_feats_zscore = node_feats
            edge_feats_zscore = edge_feats
        assert not node_feats_zscore.isnan().any()
        assert not edge_feats_zscore.isnan().any()
        graph = \
            GraphObj(
                g, rho_feats_zscore,
                node_feats_zscore,
                edge_feats_zscore,
                edge_index, self.edge_feat_cfg)
        return graph

    def _get_curve_dummy(self, graph_pn, curve_pn, zscore=True, eps=1e-10):
        with open(curve_pn, 'rb') as fp:
            c = pickle.load(fp)

        c_magnitude, c_shape = get_curve(c['curve'], **self.curve_norm_cfg)

        with open(graph_pn, 'rb') as f:
            g = pickle.load(f)

        ########################## TOY CASE ##########################
        feat_idx = EDGE_FEAT_CFG.index('l2_dist')
        c_magnitude = torch.sum(torch.FloatTensor(g.graph.pop('edge_feats')[:,feat_idx])).unsqueeze(0)

        if zscore:
            c_magnitude_mean, c_magnitude_std, c_shape_mean, c_shape_std = self.c_stats
            c_magnitude_zscore = (c_magnitude - c_magnitude_mean) / (c_magnitude_std + eps)
            c_shape_zscore = (c_shape - c_shape_mean) / (c_shape_std + eps)
        else:
            c_magnitude_zscore = c_magnitude
            c_shape_zscore = c_shape
        assert not c_magnitude_zscore.isnan().any()
        assert not c_shape_zscore.isnan().any()

        c['curve'][:,1] = np.arange(c_shape.shape[0])/(c_shape.shape[0]-1) * c_magnitude.item()
        curve = \
            CurveObj(c['curve'], c_magnitude_zscore, c_shape_zscore)  # TODO
        return curve

    def _get_curve(self, curve_pn, is_zscore_curve_magni=True, is_zscore_curve_shape=True, eps=1e-10):
        with open(curve_pn, 'rb') as fp:
            c = pickle.load(fp)

        if self.apply_patch:
            c['curve'][:, 1:] *= 4/7
        if self.augment_curve:
            # if type(self.augment_curve) == bool:
            r = random.random()
            c['curve'][:, 1:] += 40.0
            Z = np.expand_dims(c['curve'][:, 1:].max(axis=0),0)
            if r < 0.4:
                c['curve'][:, 1:] /= Z
                c['curve'][:, 1:] = timewarp(c['curve'][:, 1:], 0.2)
                c['curve'][:, 1:] *= Z
            elif r < 0.8:
                c['curve'][:, 1:] /= Z
                c['curve'][:, 1:] = magnitudewarp(c['curve'][:, 1:], 0.2)
                c['curve'][:, 1:] *= Z
            else:
                pass
            c['curve'][:, 1:] -= 40.0
            # else:
            #     c['curve'][:, 1:] += 40.0
            #     Z = np.expand_dims(c['curve'][:, 1:].max(axis=0),0)
            #     c['curve'][:, 1:] /= Z
            #     c['curve'][:, 1:] = timewarp(c['curve'][:, 1:], 0.2)
            #     c['curve'][:, 1:] = magnitudewarp(c['curve'][:, 1:], 0.2)
            #     c['curve'][:, 1:] *= Z
            #     c['curve'][:, 1:] -= 40.0

        if self.digitize_cfg is not None:
            digital_curve = \
                self.digitize_curve_np(
                    np.expand_dims(c['curve'][:, 1], axis=0))[0]
            c['curve'] = c['curve'][:self.digitize_cfg['n_freq']]
            c['curve'][:, 1] = digital_curve.astype(int)


        if ETH_FULL_C_VECTOR:
            c['curve'] = torch.cat((torch.tensor([[0],[1]]), torch.tensor(c['curve'])), dim=1)
        else:
            c['curve'][:, 1][c['curve'][:,1] < -40.0] = -40.0
        c_magnitude, c_shape = get_curve(c['curve'], **self.curve_norm_cfg)

        if is_zscore_curve_magni:
            c_magnitude_mean, c_magnitude_std, c_shape_mean, c_shape_std = self.c_stats
            c_magnitude_zscore = (c_magnitude - c_magnitude_mean) / (c_magnitude_std + eps)
        else:
            c_magnitude_zscore = c_magnitude

        if is_zscore_curve_shape:
            c_magnitude_mean, c_magnitude_std, c_shape_mean, c_shape_std = self.c_stats
            c_shape_zscore = (c_shape - c_shape_mean) / (c_shape_std + eps)
        else:
            c_shape_zscore = c_shape
        assert not c_magnitude_zscore.isnan().any()
        assert not c_shape_zscore.isnan().any()

        if self.binning_dict is None:
            weight_magnitude = 1.0
            weight_shape = 1.0
        else:
            weight_magnitude = \
                bins2weight(
                    self.binning_dict['data_magnitude'], c['cid'],
                    self.binning_dict['num_classes_magnitude'])
            weight_shape = \
                bins2weight(
                    self.binning_dict['data_shape'], c['cid'],
                    self.binning_dict['num_classes_shape'])

        curve = \
            CurveObj(c['curve'], c_magnitude_zscore, c_shape_zscore, weight_magnitude, weight_shape)
        return curve

    def get_unnormalized_curve_from_cid(self, cid):
        curve_pn = self.cid2curve_pn[cid]
        with open(curve_pn, 'rb') as fp:
            c = pickle.load(fp)
        c['curve'][:, 1][c['curve'][:,1] < -40.0] = -40.0
        return c['curve'][:, 1]

    def digitize_curve_np(self, curve):
        '''
        Args:
            curve: 2xL curve
        Returns:
            curve_digitized: 2xn_freq curve
        '''
        bins = np.array([float(x) for x in self.digitize_cfg['bins']])
        n_freq = self.digitize_cfg['n_freq']
        blocksize = int(curve.shape[1] / n_freq)
        assert curve.shape[1] % n_freq == 0
        out = np.median(np.digitize(curve, bins).reshape(curve.shape[0],-1,blocksize),axis=-1).astype(int)
        # curve[:n_freq, 1] = \
        #     block_reduce(np.digitize(curve[:, 1], bins), block_size=blocksize, func=np.median)
        # curve[:n_freq, 0] = \
        #     block_reduce(curve[:, 0], block_size=blocksize, func=np.median)
        return out

    def digitize_curve_torch(self, curve):
        '''
        Args:
            curve: BxL curve
        Returns:
            curve_digitized: Bxn_freq curve
        '''
        bins = torch.Tensor([float(x) for x in self.digitize_cfg['bins']]).to(curve.device)
        n_freq = self.digitize_cfg['n_freq']
        blocksize = int(curve.shape[1] / n_freq)
        # out = \
        #     torch.median(torch.bucketize(curve, bins).reshape(
        #         curve.shape[0], -1, blocksize), dim=-1)[0].long()
        out = torch.mean(curve.reshape(curve.shape[0], -1, blocksize), dim=-1)
        return out

    def unnormalize_curve(self, c_magnitude_zscore, c_shape_zscore, c_magnitude_u=None, c_shape_u=None):
        c_magnitude_mean, c_magnitude_std, c_shape_mean, c_shape_std = self.c_stats

        device = c_magnitude_zscore.device
        c_magnitude_std, c_magnitude_mean = c_magnitude_std.to(device), c_magnitude_mean.to(device)
        c_shape_std, c_shape_mean = c_shape_std.to(device), c_shape_mean.to(device)

        if self.is_zscore_curve_magni is None or \
           type(self.is_zscore_curve_magni) == bool and not self.is_zscore_curve_magni:
            c_magnitude = c_magnitude_zscore
        elif type(self.is_zscore_curve_magni) == bool and self.is_zscore_curve_magni:
            c_magnitude = c_magnitude_zscore * (c_magnitude_std+1e-10) + c_magnitude_mean
        elif type(self.is_zscore_curve_magni) == str and self.is_zscore_curve_magni == 'minmax':
            assert self.curve_norm_cfg['curve_method'] is None
            c_magnitude = c_magnitude_zscore
        else:
            assert False

        if ETH_FULL_C_VECTOR:
            c_shape = c_shape_mean
        elif self.is_zscore_curve_shape is None or \
           type(self.is_zscore_curve_shape) == bool and not self.is_zscore_curve_shape:
            c_shape = c_shape_zscore
        elif type(self.is_zscore_curve_shape) == bool and self.is_zscore_curve_shape:
            c_shape = c_shape_zscore * (c_shape_std+1e-10) + c_shape_mean
        elif type(self.is_zscore_curve_shape) == str and self.is_zscore_curve_shape == 'minmax':
            assert self.curve_norm_cfg['curve_method'] is None
            c_shape = c_shape_zscore * (c_shape_std+1e-10) + c_shape_mean
        else:
            assert False

        if args['forward_model']['loss_coeff']['shape_coeff'] is None:
            c_magnitude = \
                DEFAULT_MAGNITUDE * torch.ones(c_magnitude.shape, dtype=torch.float, device=c_magnitude.device)
            # assert False, f'should only hit this with PSU dataset.'
            print(f'WARNING: should only hit this with PSU dataset.')
        c, cmin, cmax = unnormalize_curve(c_magnitude, c_shape, **self.curve_norm_cfg)

        if c_magnitude_u is None or c_shape_u is None:
            c_UBLB = None
        else:
            c_magnitude_UB = \
                (c_magnitude_zscore+c_magnitude_u) * (c_magnitude_std+1e-10) + c_magnitude_mean

            if self.is_zscore_curve_shape is None or \
                    type(self.is_zscore_curve_shape) == bool and not self.is_zscore_curve_shape:
                c_shape_UB = c_shape_zscore+c_shape_u
            elif type(self.is_zscore_curve_shape) == bool and self.is_zscore_curve_shape:
                c_shape_UB = \
                    (c_shape_zscore+c_shape_u) * (c_shape_std+1e-10) + c_shape_mean
            else:
                assert False

            c_UB, *_ = \
                unnormalize_curve(
                    c_magnitude_UB, c_shape_UB,
                    cmin=cmin, cmax=cmax, **self.curve_norm_cfg)

            c_magnitude_LB = \
                (c_magnitude_zscore-c_magnitude_u) * (c_magnitude_std+1e-10) + c_magnitude_mean

            if self.is_zscore_curve_shape is None or \
                    type(self.is_zscore_curve_shape) == bool and not self.is_zscore_curve_shape:
                c_shape_LB = c_shape_zscore-c_shape_u
            elif type(self.is_zscore_curve_shape) == bool and self.is_zscore_curve_shape:
                c_shape_LB = \
                    (c_shape_zscore-c_shape_u) * (c_shape_std+1e-10) + c_shape_mean
            else:
                assert False

            c_LB, *_ = \
                unnormalize_curve(
                    c_magnitude_LB, c_shape_LB,
                    cmin=cmin, cmax=cmax, **self.curve_norm_cfg)

            c_UBLB = (c_UB, c_LB)

        return c, c_UBLB


class GraphCurveDatasetContrastive(GraphCurveDataset):
    def __init__(self, graph_pn, curve_pn, mapping_pn, split, is_zscore_graph,
                 is_zscore_curve_magni, is_zscore_curve_shape,
                 node_feat_cfg, edge_feat_cfg, curve_norm_cfg, digitize_cfg=None,
                 augment_graph=False, augment_curve=False, g_stats=None, c_stats=None,
                 use_cosine_node=0, use_cosine_edge=0, binning_dict=None):
        super().__init__(graph_pn, curve_pn, mapping_pn, split, is_zscore_graph,
                 is_zscore_curve_magni, is_zscore_curve_shape,
                 node_feat_cfg, edge_feat_cfg, curve_norm_cfg, digitize_cfg=digitize_cfg,
                 augment_graph=augment_graph, augment_curve=augment_curve, g_stats=g_stats, c_stats=c_stats,
                 use_cosine_node=use_cosine_node, use_cosine_edge=use_cosine_edge, binning_dict=binning_dict)
        self.cid2gid = {}
        self.cid_li = []
        for gid, cid in self.mapping.gid_cid_li:
            self.cid_li.append(cid)
            self.cid2gid[cid] = gid
        self.gid_map_positive = self.get_gid_map_positive()

    def get_gid_map_positive(self):
        all_curves = torch.stack([self._get_curve_wrapper(cid) for cid in self.cid_li], dim=0)
        gid_map_positive = {}
        print('forming contrastive pairs')
        for cid in tqdm(self.cid_li):
            gid = self.cid2gid[cid]
            gid_map_positive[gid] = self.find_kmost_similar_gid(cid, all_curves)
        return gid_map_positive

    def find_kmost_similar_gid(self, cid, all_curves, k=3):
        curve_cid = self._get_curve_wrapper(cid).unsqueeze(dim=0)
        metric = -torch.sum((curve_cid - all_curves)**2, dim=-1) # assert metric.shape == (-1,)
        _, arg_topk = torch.topk(metric, k=k+1)
        return [self.cid2gid[self.cid_li[idx]] for idx in arg_topk[1:]]

    def _get_curve_wrapper(self, cid):
        curve_pn = self.cid2curve_pn[cid]
        curve = self._get_curve(
            curve_pn,
            is_zscore_curve_magni=self.is_zscore_curve_magni,
            is_zscore_curve_shape=self.is_zscore_curve_shape,
        )
        return curve.c_shape.view(-1)

    def __getitem__(self, idx):
        gid, cid = self.mapping.gid_cid_li[idx]
        gid_pos = random.choice(self.gid_map_positive[gid])
        gid_neg = self.cid2gid[random.choice(self.cid_li)]
        while gid == gid_neg:
            gid_neg = self.cid2gid[random.choice(self.cid_li)]

        graph_pn = self.gid2graph_pn[gid]
        graph_pn_pos = self.gid2graph_pn[gid_pos]
        graph_pn_neg = self.gid2graph_pn[gid_neg]
        curve_pn = self.cid2curve_pn[cid]
        graph = self._get_graph(graph_pn, zscore=self.is_zscore_graph)
        graph_pos = self._get_graph(graph_pn_pos, zscore=self.is_zscore_graph)
        graph_neg = self._get_graph(graph_pn_neg, zscore=self.is_zscore_graph)
        # curve = self._get_curve_dummy(graph_pn, curve_pn, zscore=self.is_zscore_curve)
        curve = self._get_curve(
            curve_pn,
            is_zscore_curve_magni=self.is_zscore_curve_magni,
            is_zscore_curve_shape=self.is_zscore_curve_shape,
        )
        return graph, curve, graph_pos, graph_neg

    def collate_fn(self, samples):
        graph_li, curve_li, graph_pos_li, graph_neg_li = zip(*samples)
        graph_collated = GraphObjCollated.collate(graph_li)
        curve_collated = CurveObjCollated.collate(curve_li)
        graph_pos_collated = GraphObjCollated.collate(graph_pos_li)
        graph_neg_collated = GraphObjCollated.collate(graph_neg_li)
        return graph_collated, curve_collated, graph_pos_collated, graph_neg_collated

class GraphObj:
    def __init__(self, g, rho_feats, node_feats, edge_feats, edge_index, edge_feat_cfg):
        self.g = g
        self.gid = g.graph['gid']
        self.feats_rho = rho_feats
        self.feats_node = node_feats
        self.feats_edge = edge_feats
        self.edge_index = edge_index
        assert not (edge_index[0] == edge_index[1]).any()
        assert self.feats_edge.shape[0] == self.edge_index.shape[1]
        self.edge_feat_cfg = edge_feat_cfg

    @classmethod
    def init_from_g(cls, g, g_stats, node_feat_index, edge_feat_index, is_zscore, edge_feat_cfg):
        (node_feats_mean, node_feats_std,
         edge_feats_mean, edge_feats_std,
         rho_feats_mean, rho_feats_std) = g_stats
        edge_li = get_edge_li(g)
        edge_index = get_edge_index(edge_li)
        node_feats = \
            get_node_feats(
                g, edge_index, manual_args={'radius_default':float(node_feats_mean[0, -1])})
        edge_feats = \
            get_edge_feats(g, edge_li, radius_default=float(edge_feats_mean[0, -1]))

        node_feats = torch.FloatTensor(node_feats[:,node_feat_index])
        edge_feats = torch.FloatTensor(edge_feats[:,edge_feat_index])
        rho_feats = g.graph['rho']

        if is_zscore:
            node_feats_zscore = (node_feats-node_feats_mean)/(node_feats_std+1e-10)
            edge_feats_zscore = (edge_feats-edge_feats_mean)/(edge_feats_std+1e-10)
            rho_feats_zscore = (rho_feats - rho_feats_mean) / (rho_feats_std + 1e-10)
        else:
            node_feats_zscore = node_feats
            edge_feats_zscore = edge_feats
            rho_feats_zscore = rho_feats
        return cls(g, rho_feats_zscore, node_feats_zscore, edge_feats_zscore, edge_index, edge_feat_cfg)

    def to(self, device):
        self.edge_index = self.edge_index.to(device)
        self.feats_node = self.feats_node.to(device)
        self.feats_edge = self.feats_edge.to(device)
        self.feats_rho = self.feats_rho.to(device)


class GraphObjCollatedv2:
    def __init__(self, feats_node, edge_index, graph_index, graph_base_index, aid_index,
                 bid2nid2aid, mask, g_stats, node_feats_index, edge_feats_index, edge_feat_cfg):
        self.feats_node = feats_node
        self.edge_index = edge_index
        self.graph_index = graph_index # [0,...,b-1] ^ n_nodes x 1
        self.graph_base_index = graph_base_index # [0,1] ^ n_nodes x 1
        self.aid_index = aid_index # [0,...,|A|-1] ^ n_nodes x 1
        self.bid2nid2aid = bid2nid2aid
        self.node_feat_index = node_feats_index
        self.edge_feat_index = edge_feats_index
        self.mask = mask
        self.g_stats = g_stats
        self.edge_feat_cfg = edge_feat_cfg # for forward model

    @classmethod
    def from_first_nodes(cls, aid_li, g_stats, node_feats_index, edge_feats_index, edge_feat_cfg):
        feats_node = AID2FEATS[aid_li]
        edge_index = torch.empty((2,0), dtype=torch.long, device=aid_li.device)
        graph_index = torch.arange(aid_li.shape[0], device=aid_li.device)
        graph_base_index = torch.ones_like(aid_li, dtype=torch.bool, device=aid_li.device)
        aid_index = aid_li
        bid2nid2aid = defaultdict(lambda: DoubleDict())
        for bnid, aid in enumerate(aid_li):
            bid2nid2aid[bnid].add(bnid, aid.item())
        mask = torch.zeros(len(aid_li), dtype=torch.bool)
        return cls(
            feats_node, edge_index, graph_index, graph_base_index,
            aid_index, bid2nid2aid, mask, g_stats, node_feats_index, edge_feats_index, edge_feat_cfg)

    def to(self, device):
        self.feats_node = self.feats_node.to(device)
        self.edge_index = self.edge_index.to(device)
        self.graph_index = self.graph_index.to(device) # [0,...,b-1] ^ n_nodes x 1
        self.graph_base_index = self.graph_base_index.to(device) # [0,1] ^ n_nodes x 1
        self.aid_index = self.aid_index.to(device) # [0,...,|A|-1] ^ n_nodes x 1
        self.mask = self.mask.to(device)

    def get_fwd_model_graph(self, base_cell='octet', rm_redundant=True, rho=None, is_zscore=False):
        # import time
        # t0 = time.time()
        tetrahedron_li = self.get_g_li(self.feats_node, self.edge_index, self.graph_index, rho=rho)
        # print(f'get g_li: {time.time()-t0}s')
        # t0 = time.time()

        g_li = []
        r_li = []
        for tetrahedron in tetrahedron_li:
            g = tesselate(
                tetrahedron,
                start='tetrahedron',
                end=base_cell,
                rm_redundant=rm_redundant
            )

            g, _ = rm_redundant_nodes(g)
            for eid in g.edges():
                nid_start, nid_end = eid
                g.edges[eid]['length'] = \
                    np.linalg.norm(g.nodes[nid_start]['coord'] - g.nodes[nid_end]['coord'])
            r = rho2r(g.graph['rho'], g, base_cell=DATASET_CFG['base_cell'])
            for eid in g.edges():
                g.edges[eid]['radius'] = r
            g_li.append(g)
            r_li.append(r)
        # print('sdfg',rho2r(g_li[1].graph['gid'], g_li[1]))
        # print(f'tesselate: {time.time()-t0}s')
        # t0 = time.time()

        graph = GraphObjCollated.init_from_g_li(
            g_li, self.g_stats, self.node_feat_index, self.edge_feat_index, is_zscore, self.edge_feat_cfg)
        # print(f'init_from_g_li: {time.time()-t0}s')
        # print('dsf',rho2r(g_li[1].graph['gid'], g_li[1]))
        # print('asdf',rho2r(graph.g_li[1].graph['gid'], graph.g_li[1]))
        # assert False
        return graph, r_li

    def get_g_li(self, feats_node, edge_index, graph_index, rho=None):
        if rho is not None:
            rho = rho.detach().cpu().numpy()
        gid_li = list(set(graph_index.detach().cpu().tolist()))

        g = nx.Graph()
        g.add_edges_from(edge_index.transpose(0,1).detach().cpu().numpy())
        for nid, (x,y,z) in enumerate(feats_node):
            g.nodes[nid]['coord'] = np.array([x,y,z])
        assert sorted(gid_li) == list(range(len(gid_li)))
        g = compute_lengths(g)

        g_arange = np.arange(g.number_of_nodes())
        g_li = []
        for gid in sorted(gid_li):
            sg = deepcopy(g.subgraph(g_arange[(graph_index == gid).detach().cpu().numpy()]))
            sg = nx.convert_node_labels_to_integers(sg)
            sg.graph['rho'] = rho[gid] if rho is not None else None
            sg.graph['gid'] = gid
            g_li.append(sg)
        return g_li

    def add_new_edges(self, aid_start_li, aid_end_li):
        n_nodes_cur = self.feats_node.shape[0]
        offset = 0
        feats_node_to_add = []
        graph_index_to_add = []
        graph_base_index_to_change = []
        aid_index_to_add = []
        edge_index_to_add = []
        aid_bw_li, aid_mask_li = \
            get_aid_bw_li(
                aid_start_li, aid_end_li,
                [list(self.bid2nid2aid[bid].r2l.keys()) for bid in sorted(self.bid2nid2aid.keys())]
            )
        for bid, (skip, aid_start, aid_bw, aid_end, aid_mask) in \
                enumerate(zip(self.mask, aid_start_li, aid_bw_li, aid_end_li, aid_mask_li)):
            if skip:
                continue
            nid_start = self.bid2nid2aid[bid].r2l[int(aid_start)]

            collision = int(aid_end) in self.bid2nid2aid[bid].r2l
            if collision:
                nid_end = self.bid2nid2aid[bid].r2l[int(aid_end)]
            else:
                nid_end = n_nodes_cur + offset
                offset += 1

            aid_bw_valid = np.array(aid_bw)[aid_mask]
            nid_bw = n_nodes_cur + offset + np.arange(len(aid_bw_valid))
            offset += len(aid_bw_valid)

            aid_new = \
                torch.tensor(aid_bw_valid, dtype=torch.long, device=aid_start_li.device) \
                if collision else \
                torch.tensor([aid_end] + list(aid_bw_valid), dtype=torch.long, device=aid_start_li.device)

            self.bid2nid2aid[bid].add(int(nid_start), int(aid_start))
            for nid, aid in zip(nid_bw, aid_bw_valid):
                self.bid2nid2aid[bid].add(int(nid), int(aid))
            self.bid2nid2aid[bid].add(int(nid_end), int(aid_end))

            feats_node_to_add.append(AID2FEATS[aid_new]) # TODO
            graph_index_to_add.append(bid*torch.ones_like(aid_new, device=aid_start_li.device))
            graph_base_index_to_change.append(nid_end)
            aid_index_to_add.append(aid_new)
            edge_index_to_add.append(
                torch.tensor(
                    [[nid_start] + list(nid_bw), list(nid_bw) + [nid_end]],
                    dtype=torch.long, device=aid_start_li.device)
            )

        assert len(graph_index_to_add) == len(feats_node_to_add) == len(aid_index_to_add)
        if len(feats_node_to_add) > 0:
            self.feats_node = torch.cat([self.feats_node] + feats_node_to_add, dim=0)

        if len(graph_index_to_add) > 0:
             self.graph_index = torch.cat([self.graph_index] + graph_index_to_add, dim=0)
             self.graph_base_index = \
                 torch.cat(
                     [self.graph_base_index] + [
                         torch.zeros_like(x, dtype=torch.bool, device=aid_start_li.device) \
                         for x in graph_index_to_add
                     ], dim=0)

        if len(graph_base_index_to_change) > 0:
            self.graph_base_index[graph_base_index_to_change] = True

        if len(aid_index_to_add) > 0:
            self.aid_index = torch.cat([self.aid_index] + aid_index_to_add, dim=0)

        if len(edge_index_to_add) > 0:
            self.edge_index = torch.cat([self.edge_index] + edge_index_to_add, dim=1)

        return 0

class CurveOverideDataset(GraphCurveDataset):
    def __init__(self, curve_li, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curve_li = curve_li

    def __len__(self):
        return len(self.curve_li)

    def __getitem__(self, idx):
        gid, _ = self.mapping.gid_cid_li[0]
        graph_pn = self.gid2graph_pn[gid]
        c_curve = self.curve_li[idx]
        graph = self._get_graph(graph_pn, zscore=self.is_zscore_graph)
        curve = self._get_curve_custom(
            c_curve,
            is_zscore_curve_magni=self.is_zscore_curve_magni,
            is_zscore_curve_shape=self.is_zscore_curve_shape,
        )
        return graph, curve

    def _get_curve_custom(self, c_curve, is_zscore_curve_magni, is_zscore_curve_shape, eps=1e-10):
        # c_magnitude_zscore is not used!
        c_curve = np.stack((np.arange(len(c_curve))/(len(c_curve)-1), c_curve), axis=1)

        assert args['dataset']['curve_norm_cfg']['curve_method'] is None
        if self.digitize_cfg is not None:
            digital_curve = \
                self.digitize_curve_np(
                    np.expand_dims(c_curve[:, 1], axis=0))[0]
            c_curve = c_curve[:self.digitize_cfg['n_freq']]
            c_curve[:, 1] = digital_curve.astype(int)

        c_curve[:, 1][c_curve[:,1] < -40.0] = -40.0
        c_magnitude, c_shape = get_curve(c_curve, **self.curve_norm_cfg)

        if is_zscore_curve_magni:
            c_magnitude_mean, c_magnitude_std, c_shape_mean, c_shape_std = self.c_stats
            c_magnitude_zscore = (c_magnitude - c_magnitude_mean) / (c_magnitude_std + eps)
        else:
            c_magnitude_zscore = c_magnitude

        if is_zscore_curve_shape:
            c_magnitude_mean, c_magnitude_std, c_shape_mean, c_shape_std = self.c_stats
            c_shape_zscore = (c_shape - c_shape_mean) / (c_shape_std + eps)
        else:
            c_shape_zscore = c_shape
        assert not c_magnitude_zscore.isnan().any()
        assert not c_shape_zscore.isnan().any()

        assert self.binning_dict is None
        weight_magnitude = 1.0
        weight_shape = 1.0

        curve = \
            CurveObj(c_curve, c_magnitude_zscore, c_shape_zscore, weight_magnitude, weight_shape)
        return curve

class GraphObjCollated:
    def __init__(self, g_li, edge_index, feats_rho, feats_node, feats_edge,
                 graph_node_index, graph_edge_index, edge_feat_cfg):
        self.g_li = g_li
        self.feats_rho = feats_rho
        self.feats_node = feats_node
        self.feats_edge = feats_edge
        self.edge_index = edge_index
        # [0, ..., 0, 1, ..., 1, 2, ..., 2] denoting graph each node belongs to
        self.graph_node_index = graph_node_index
        # [0, ..., 0, 1, ..., 1, 2, ..., 2] denoting graph each edge belongs to
        self.graph_edge_index = graph_edge_index
        self.edge_feat_cfg = edge_feat_cfg

    def update_radius(self, radius_li, rho, is_zscore, g_stats):
        if 'radius' in self.edge_feat_cfg:
            feat_idx = self.edge_feat_cfg.index('radius')
            _, _, edge_feats_mean, edge_feats_std, rho_feats_mean, rho_feats_std = g_stats

            for i, r in enumerate(radius_li):
                if is_zscore:
                    f = (r-edge_feats_mean[0,feat_idx])/(edge_feats_std[0,feat_idx]+1e-10)
                else:
                    f = r
                self.feats_edge[self.graph_edge_index == i, feat_idx] = f
        # if is_zscore:
        #     feats_rho = \
        #         (torch.tensor(rho).to(
        #             self.feats_rho.device).reshape(*self.feats_rho.shape)
        #          - rho_feats_mean) / rho_feats_std
        # else:
        feats_rho = \
            torch.tensor(rho).to(self.feats_rho.device).reshape(*self.feats_rho.shape)
        self.feats_rho = feats_rho

    def __getitem__(self, idx):
        # self.graph_edge_index.shape, self.graph_node_index.shape
        offset = sum([g.number_of_nodes() for g in self.g_li[:idx]])
        g = self.g_li[idx]
        feats_rho = self.feats_rho[idx]
        feats_node = self.feats_node[self.graph_node_index == idx]
        feats_edge = self.feats_edge[self.graph_edge_index == idx]
        edge_index = self.edge_index[:,self.graph_edge_index == idx] - offset
        return GraphObj(g, feats_rho, feats_node, feats_edge, edge_index, self.edge_feat_cfg)

    @classmethod
    def collate(cls, graph_li):
        offset = 0
        g_li = []
        edge_index = []
        feats_rho = []
        feats_node = []
        feats_edge = []
        graph_node_index = []
        graph_edge_index = []
        edge_feat_cfg = None
        for i, graph in enumerate(graph_li):
            if edge_feat_cfg is None:
                edge_feat_cfg = graph.edge_feat_cfg
            else:
                assert edge_feat_cfg == graph.edge_feat_cfg
            g_li.append(graph.g)
            edge_index.append(graph.edge_index.detach().clone()+offset)
            feats_rho.append(graph.feats_rho)
            feats_node.append(graph.feats_node)
            feats_edge.append(graph.feats_edge)
            graph_node_index.extend([i]*graph.g.number_of_nodes())
            graph_edge_index.extend([i]*(2*graph.g.number_of_edges()))
            offset += graph.g.number_of_nodes()
            # assert graph.feats_node.shape[0] == graph.g.number_of_nodes()
            # assert graph.feats_edge.shape[0] == 2*graph.g.number_of_edges()
        feats_rho = torch.tensor(feats_rho)
        edge_index = torch.cat(edge_index, dim=1)
        feats_node = torch.cat(feats_node, dim=0)
        feats_edge = torch.cat(feats_edge, dim=0)
        graph_node_index = torch.tensor(graph_node_index, dtype=torch.long)
        graph_edge_index = torch.tensor(graph_edge_index, dtype=torch.long)
        return cls(g_li, edge_index, feats_rho, feats_node, feats_edge, graph_node_index, graph_edge_index, edge_feat_cfg)

    def to(self, device):
        self.edge_index = self.edge_index.to(device)
        self.feats_node = self.feats_node.to(device)
        self.feats_edge = self.feats_edge.to(device)
        self.feats_rho = self.feats_rho.to(device)
        self.graph_node_index = self.graph_node_index.to(device)
        self.graph_edge_index = self.graph_edge_index.to(device)

    @classmethod
    def init_from_g_li(cls, g_li, g_stats, node_feat_index, edge_feat_index, is_zscore, edge_feat_cfg):
        return cls.collate([
            GraphObj.init_from_g(g, g_stats, node_feat_index, edge_feat_index, is_zscore, edge_feat_cfg)
            for g in g_li])

class CurveObj:
    def __init__(self, c, c_magnitude, c_shape, weight_magnitude=1.0, weight_shape=1.0):
        self.c = c
        self.c_magnitude = c_magnitude.type(torch.float)
        self.c_shape = c_shape.type(torch.float)
        self.weight_magnitude = weight_magnitude
        self.weight_shape = weight_shape

    def to(self, device):
        self.c_magnitude = self.c_magnitude.to(device)
        self.c_shape = self.c_shape.to(device)


class CurveObjCollated:
    def __init__(self, c_li, c_magnitude, c_shape, weight_shape, weight_magnitude):
        self.c_li = c_li
        self.c_magnitude = c_magnitude
        self.c_shape = c_shape
        self.weight_shape = weight_shape
        self.weight_magnitude = weight_magnitude

    def __len__(self):
        return len(self.c_li)

    def __getitem__(self, idx):
        c = self.c_li[idx]
        c_magnitude = self.c_magnitude[idx]
        c_shape = self.c_shape[idx]
        weight_shape = self.weight_shape[idx]
        weight_magnitude = self.weight_magnitude[idx]
        return CurveObj(c, c_magnitude, c_shape, weight_magnitude, weight_shape)

    @classmethod
    def collate(cls, curve_li):
        c_li = []
        c_magnitude = []
        c_shape = []
        for curve in curve_li:
            c_li.append(curve.c)
            c_magnitude.append(curve.c_magnitude)
            c_shape.append(curve.c_shape)
        c_magnitude = torch.stack(c_magnitude, dim=0)
        c_shape = torch.stack(c_shape, dim=0)
        weight_magnitude = torch.tensor([c.weight_magnitude for c in curve_li])
        weight_shape = torch.tensor([c.weight_shape for c in curve_li])
        return cls(c_li, c_magnitude, c_shape, weight_magnitude, weight_shape)

    def to(self, device):
        self.c_magnitude = self.c_magnitude.to(device)
        self.c_shape = self.c_shape.to(device)


class MappingObj:
    def __init__(self, pn_tsv):
        gid_cid_li = []
        with open(pn_tsv) as fp:
            tsv_file = csv.reader(fp, delimiter="\t")
            for i, (gid, cid) in enumerate(tsv_file):
                gid, cid = int(gid), int(cid)
                gid_cid_li.append((gid, cid))
        # cutoff = int(len(gid_cid_li)/100)
        # self.gid_cid_li = gid_cid_li[:cutoff]
        self.gid_cid_li = gid_cid_li

        gid2cid_li = defaultdict(list)
        cid2gid_li = defaultdict(list)
        for gid, cid in gid_cid_li:
            gid2cid_li[gid].append(cid)
            cid2gid_li[cid].append(gid)
        self.gid2cid_li = gid2cid_li
        self.cid2gid_li = cid2gid_li
