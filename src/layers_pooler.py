import torch
import torch.nn as nn

from torch_scatter import scatter_add, scatter_softmax, scatter_max
from src.utils import MLP

# pooler = Pooler(dim_in=dim, dim_out=dim, **pooler_cfg)
class Pooler(nn.Module):
    def __init__(self, dim, pool_name):
        super(Pooler, self).__init__()
        self.pool_nodes = STR2POOL[pool_name](dim)
        self.pool_edges = STR2POOL[pool_name](dim)

    def forward(self, emb_nodes, emb_edges, graph):
        emb_nodes_pooled = \
            self.pool_nodes(
                src=emb_nodes, index=graph.graph_node_index,
                dim=0, dim_size=len(graph.g_li)
            )
        emb_edges_pooled = \
            self.pool_edges(
                src=emb_edges, index=graph.graph_edge_index,
                dim=0, dim_size=len(graph.g_li)
            )
        return emb_nodes_pooled, emb_edges_pooled

class Merger(nn.Module):
    def __init__(self, dim_in, dim_out, merge_name, num_inputs=2):
        super(Merger, self).__init__()
        self.merger = STR2MERGER[merge_name](dim_in, dim_out, num_inputs)
        self.bn = nn.BatchNorm1d(dim_out)
        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self, *emb_li):
        emb_all = self.merger(*emb_li)
        emb_all = self.bn(emb_all)
        return emb_all

class CatMerger(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_inputs=2):
        super(CatMerger, self).__init__()
        self.mlp = nn.Linear(num_inputs * dim_in, dim_hidden)

    def forward(self, *emb_li):
        return self.mlp(torch.cat(emb_li, dim=-1))

class GatedAttentionPooling(nn.Module):
    def __init__(self, dim_in):
        super(GatedAttentionPooling, self).__init__()
        self.mlp_att = MLP([dim_in, dim_in//2, dim_in//4, 1])

    def forward(self, src, index, dim, dim_size):
        att = scatter_softmax(self.mlp_att(src).view(-1), index=index).unsqueeze(-1)
        Z = scatter_add(src=att*src, index=index, dim=dim, dim_size=dim_size)
        return Z

STR2POOL = {'SUM': (lambda x: scatter_add), 'ATT': GatedAttentionPooling, 'MAX': (lambda x: scatter_max)}
STR2MERGER = {'CAT': CatMerger}