# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 15:46:40 2023

@author: marco
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch_geometric.nn.conv import GATConv
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from src.generative_graph.env_v2 import NODES, EDGES, RADIUS
from src.dataset import ACTION_SPACE
from src.layers_pooler import Pooler
from src.layers_encoder import GNNEncoder
from src.layers_decoder_magnitude import MLPWrapper
from src.config import args

device = args['device']

class ReadoutNode(torch.nn.Module):
    def __init__(self, emb_dim, output_ids):
        super(ReadoutNode, self).__init__()
        assert sorted(list(output_ids)) == list(range(len(output_ids))) # for readout nodes
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ELU(),
            nn.Linear(emb_dim, emb_dim//2), nn.ELU(),
            nn.Linear(emb_dim//2, emb_dim//4), nn.ELU(),
            nn.Linear(emb_dim//4, len(output_ids))
        )

    def forward(self, x_state, output_ids):
        logits_node = self.mlp(x_state)[output_ids].reshape(-1)
        return logits_node

class ReadoutEdge(torch.nn.Module):
    def __init__(self, emb_dim):
        super(ReadoutEdge, self).__init__()
        self.mha = nn.MultiheadAttention(emb_dim, num_heads=4, batch_first=True)
        self.bilin_mat = torch.nn.Parameter(torch.rand((emb_dim, emb_dim, emb_dim)), requires_grad=True)
        self.readout = nn.Sequential(
            nn.Linear(emb_dim, emb_dim//2), nn.ELU(),
            nn.Linear(emb_dim//2, emb_dim//4), nn.ELU(),
            nn.Linear(emb_dim//4, 1)
        )

    def forward(self, x_state, x_edge):
        # compute edge probabilities
        x_edge, _ = self.mha(x_edge, x_state, x_state)
        sim_mat = torch.einsum('ij,jkl,hk->ihl', x_edge, self.bilin_mat, x_edge)
        logits_edge = self.readout(sim_mat).view(*sim_mat.shape[:2])
        return logits_edge

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class SimpleEncoder(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

class TransformerModel(torch.nn.Module):
    def __init__(self, dim, n_freq, n_bins, nhead=8, nlayers=3):
        super().__init__()
        self.dim = dim
        self.n_freq = n_freq
        self.n_bins = n_bins

        # self.pos_encoder = PositionalEncoding(dim, max_len=n_freq+1)
        # encoder_layers = TransformerEncoderLayer(dim, nhead, dim)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.embedding = nn.Embedding(n_freq*n_bins+1, dim)

        self.transformer_encoder = nn.Sequential(
            nn.Conv1d(8, 32, 4, stride=2), # (16-4)/2+1=7
            nn.Conv1d(32, 128, 4), # 7-4+1 = 4
            nn.Conv1d(128, 256, 4)) # 4-4+1 = 1
        # self.transformer_encoder = nn.Sequential(
        #     nn.Conv1d(8, 32, 4, stride=2), # (128-4)/2+1=63
        #     nn.Conv1d(32, 128, 5, stride=2), # (63-5)/2+1=30
        #     nn.Conv1d(128, 256, 4, stride=2), # (30-4)/2+1=14
        #     nn.AvgPool1d(14, 1, 0)) # (14-14)/1+1 = 1
        self.embedding = nn.Embedding(n_bins, 8)

        self.bos_token = n_freq*n_bins
        # self.batch_norm = nn.BatchNorm1d(dim)
        self.template = torch.arange(n_freq)*n_bins

    def forward(self, curve_shape):
        '''
        Args: curve_shape: Bx1xL
        Returns: out: BxDx1
        '''
        assert curve_shape.shape[1] == 1
        curve_shape = curve_shape[:,0,:].transpose(0,1)
        # src = torch.cat((
        #     self.bos_token*torch.ones_like(curve_shape[[0]], dtype=torch.long, device=curve_shape.device),
        #     self.template.to(curve_shape.device).unsqueeze(-1)*curve_shape.long()), dim=0)
        # src = self.embedding(src) * math.sqrt(self.dim)
        # src = self.pos_encoder(src)
        # output = self.transformer_encoder(src)[0].unsqueeze(-1)

        src = curve_shape.long()
        src = self.embedding(src) * math.sqrt(self.dim)
        output = self.transformer_encoder(src.permute(1,2,0))

        # output = self.batch_norm(output)
        return output

class CurveEncoder(torch.nn.Module):
    def __init__(self, dim, digitize_cfg, c_stats, mlp_magnitude_cfg, cnn_shape_cfg, mlp_shape_magnitude_cfg):
        super(CurveEncoder, self).__init__()
        self.ins = mlp_shape_magnitude_cfg.pop('ins')
        if self.ins == 'both':
            self.encoder_magnitude = nn.Sequential(
                nn.Linear(1, dim),
                MLPWrapper(dim, int(dim/2), **mlp_magnitude_cfg)
            )
            dim_shape = int(dim/2)
            self.mlp_shape_magnitude = MLPWrapper(dim, dim, **mlp_shape_magnitude_cfg)
        elif self.ins == 'shape':
            dim_shape = dim
            self.mlp_shape_magnitude = MLPWrapper(dim, dim, **mlp_shape_magnitude_cfg)
        else:
            assert False

        if digitize_cfg is None:
            layers = []
            layers.extend([
                nn.Conv1d(in_channels=1, out_channels=int(dim_shape / 8), kernel_size=5, padding=2, stride=1),
                nn.MaxPool1d(4)
            ])
            if cnn_shape_cfg['use_norm']:
                layers.append(nn.BatchNorm1d(int(dim_shape/8)))
            layers.extend([
                nn.ELU(),
                nn.Conv1d(in_channels=int(dim_shape/8), out_channels=int(dim_shape/4), kernel_size=5, padding=2, stride=1),
                nn.MaxPool1d(4)
            ])
            if cnn_shape_cfg['use_norm']:
                layers.append(nn.BatchNorm1d(int(dim_shape/4)))
            layers.extend([
                nn.ELU(),
                nn.Conv1d(in_channels=int(dim_shape/4), out_channels=int(dim_shape/2), kernel_size=5, padding=2, stride=1),
                nn.MaxPool1d(4)
            ])
            if cnn_shape_cfg['use_norm']:
                layers.append(nn.BatchNorm1d(int(dim_shape/2)))
            layers.extend([
                nn.ELU(),
                nn.Conv1d(in_channels=int(dim_shape/2), out_channels=dim_shape, kernel_size=5, padding=2, stride=1),
                nn.MaxPool1d(4)
            ])
            if cnn_shape_cfg['use_norm']:
                layers.append(nn.BatchNorm1d(dim_shape))
            self.encoder_shape = nn.Sequential(*layers)
        else:
            n_bins = len([float(x) for x in digitize_cfg['bins']]) + 1
            n_freq = digitize_cfg['n_freq']
            self.encoder_shape = TransformerModel(dim, n_freq, n_bins)
        self.c_stats = c_stats

        assert dim % 2 == 0
        self.dim = dim

    def forward(self, curve_magnitude, curve_shape):
        bs, *_ = curve_shape.shape
        curve_magnitude = curve_magnitude.to(torch.device(device))
        curve_shape = curve_shape.to(torch.device(device))
        out_shape = self.encoder_shape(curve_shape.transpose(-2, -1)).squeeze(-1)

        # m_mean, m_std, *_ = self.c_stats
        # m2 = curve_shape.max(dim=1)[0]
        # a = torch.log10(m2*torch.pow(10, curve_magnitude * m_std.to(curve_shape.device) + m_mean.to(curve_shape.device)))+1
        # curve_shape = curve_shape/m2.unsqueeze(1)
        if self.ins == 'both':
            out_magnitude = self.encoder_magnitude(curve_magnitude).view(bs,int(self.dim/2))
            out = self.mlp_shape_magnitude(torch.cat([out_magnitude, out_shape], dim=-1))
        elif self.ins == 'shape':
            out = self.mlp_shape_magnitude(out_shape)
        else:
            assert False
        return out

class PolicyNetwork(torch.nn.Module):
    def __init__(
            self, gnn_encoder, curve_encoder, mlp_state_action,
            readout_stop, readout_start, readout_end, readout_rho,
            default_emb, search_cfg):
        super(PolicyNetwork, self).__init__()
        self.action_embs = nn.Parameter(torch.normal(torch.ones(ACTION_SPACE.shape[0], default_emb.shape[-1])), requires_grad=True)
        self.gnn_encoder = gnn_encoder
        self.curve_encoder = curve_encoder
        self.mlp_state_action = mlp_state_action
        self.readout_stop = readout_stop
        self.readout_start = readout_start
        self.readout_end = readout_end
        self.readout_rho = readout_rho
        self.default_emb = default_emb
        self.search_cfg = search_cfg

        # self.action_table = nn.Embedding(len(aid2action_node), emb_dim)
        self.num_aid_total = None
        self.c_stats = None

    @classmethod
    def init_from_cfg(cls, dataset, encoder_graph_cfg, encoder_curve_cfg,
                      dim, search_cfg, mlp_state_action_cfg, readout_stop_cfg,
                      readout_start_cfg, readout_end_cfg, readout_rho_cfg, **kwargs):
        dim_input_nodes = dataset.dataset.dim_input_nodes
        # dim_input_edges = dataset.dataset.dim_input_edges

        encoder = GNNEncoder(
            dim_input_nodes=dim_input_nodes, dim_input_edges=1, dim_hidden=dim, **encoder_graph_cfg)
        # pooler = Pooler(dim_in=dim, dim_out=dim, **pooler_cfg)
        curve_encoder = CurveEncoder(dim, dataset.dataset.digitize_cfg, dataset.dataset.c_stats, **encoder_curve_cfg)
        mlp_state_action = MLPWrapper(2*dim, dim, **mlp_state_action_cfg)
        readout_stop = MLPWrapper(2*dim, 1, **readout_stop_cfg)
        # readout_start = MLPWrapper(3*dim, 1, **readout_start_cfg)
        readout_start = MLPWrapper(2*dim, ACTION_SPACE.shape[0], **readout_start_cfg)
        # readout_end = MLPWrapper(4*dim, 1, **readout_end_cfg)
        readout_end = MLPWrapper(3*dim, ACTION_SPACE.shape[0], **readout_end_cfg)
        readout_rho = MLPWrapper(2*dim, 2, **readout_rho_cfg)
        default_emb = nn.Parameter(torch.randn(1, dim), requires_grad=True)
        return cls(
            encoder, curve_encoder, mlp_state_action,
            readout_stop, readout_start, readout_end, readout_rho,
            default_emb, search_cfg)

    def get_embeddings(self, state):
        action_emb = self.action_embs.unsqueeze(0).repeat(len(state), 1, 1)

        curve = state.curve_collated
        condition_emb = \
            self.curve_encoder(curve.c_magnitude, curve.c_shape)

        graph = state.graph_collated
        if graph is None:
            node_emb_state = torch.zeros_like(action_emb[:,0,:], device=action_emb.device)
            state_emb = torch.zeros_like(action_emb, device=action_emb.device)
        else:
            feats_node = graph.feats_node.to(action_emb.device)
            edge_index = graph.edge_index.to(action_emb.device)
            node_emb_state, _ = self.gnn_encoder(feats_node, None, edge_index)
            new_indices = graph.graph_index * action_emb.shape[1] + graph.aid_index
            state_emb = \
                scatter_max(
                    node_emb_state, new_indices, dim=0,
                    dim_size=action_emb.shape[0]*action_emb.shape[1]
                )[0].reshape(action_emb.shape[0], action_emb.shape[1], node_emb_state.shape[-1])

        state_action_condition_emb = \
            torch.cat((
                state_emb, action_emb, condition_emb.unsqueeze(1).repeat(1, action_emb.shape[1], 1)), dim=-1)
        embeddings = {
            'state_emb':state_emb,
            'action_emb':action_emb,
            'condition_emb':condition_emb,
            'state_action_condition_emb':state_action_condition_emb,
            'node_emb_state':node_emb_state
        }
        return embeddings

    def get_rho(self, state, node_emb_state=None, condition_emb=None, eps=1e-5, **kwargs):
        # create stop logits
        graph = state.graph_collated
        state_emb = scatter_add(node_emb_state, graph.graph_index, dim=0, dim_size=len(state))

        rho_mean, rho_std = \
            self.readout_rho(torch.cat((state_emb, condition_emb), dim=-1)).transpose(0, -1)
        # slack = 0.02
        # rho_mean = \
        #     self.search_cfg.constraint_rho_min-slack + \
        #     (self.search_cfg.constraint_rho_max - self.search_cfg.constraint_rho_min + 2*slack) * \
        #     F.sigmoid(rho_mean)
        rho_std = \
            torch.clamp(
                F.sigmoid(rho_std) * (self.search_cfg.constraint_rho_max - self.search_cfg.constraint_rho_min),
                min=eps)
        # rho_std = \
        #     (self.search_cfg.constraint_rho_max - self.search_cfg.constraint_rho_min)/4
        rho_logits = torch.distributions.normal.Normal(rho_mean, 0.05)
        return rho_logits

    def get_stop_token(self, state, node_emb_state=None, condition_emb=None, **kwargs):
        # create stop logits
        graph = state.graph_collated
        if graph is None:
            state_emb = node_emb_state
        else:
            state_emb, _ = scatter_max(node_emb_state, graph.graph_index, dim=0, dim_size=len(state))

        stop_logits = \
            self.readout_stop(torch.cat((state_emb, condition_emb), dim=-1))
        # return torch.ones_like(stop_logits, device=stop_logits.device, dtype=torch.float)
        return stop_logits # stop_logits[0][0] * torch.ones_like(stop_logits)

    def get_aid_start(self, state,
                      node_emb_state=None, condition_emb=None,
                      state_action_condition_emb=None, **kwargs):
        # start_logits = \
        #     self.readout_start(state_action_condition_emb)

        graph = state.graph_collated
        if graph is None:
            state_emb = node_emb_state
        else:
            state_emb, _ = scatter_max(node_emb_state, graph.graph_index, dim=0, dim_size=len(state))
        start_logits = \
            self.readout_start(torch.cat((state_emb, condition_emb), dim=-1)).unsqueeze(-1)
        # return torch.ones_like(start_logits, device=start_logits.device, dtype=torch.float)
        return start_logits # start_logits[0][0] * torch.ones_like(start_logits)

    def get_aid_end(self, state, aid_selected,
                    node_emb_state=None, condition_emb=None,
                    state_action_condition_emb=None, state_emb=None, **kwargs):
        # end_logits = \
        #     self.readout_end(torch.cat((
        #         state_action_condition_emb,
        #         state_emb[
        #             torch.arange(len(aid_selected)),
        #             aid_selected
        #         ].unsqueeze(1).repeat(1,state_action_condition_emb.shape[1],1)), dim=-1))

        u_emb = state_emb[torch.arange(len(aid_selected)),aid_selected]

        graph = state.graph_collated
        if graph is None:
            state_emb = node_emb_state
        else:
            state_emb, _ = scatter_max(node_emb_state, graph.graph_index, dim=0, dim_size=len(state))

        end_logits = \
            self.readout_end(torch.cat((state_emb, condition_emb, u_emb), dim=-1)).unsqueeze(-1)
        # return torch.ones_like(end_logits, device=end_logits.device, dtype=torch.float)
        return end_logits # end_logits[0][0] * torch.ones_like(end_logits)
