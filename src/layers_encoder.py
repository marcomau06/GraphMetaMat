import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from torch_geometric.nn import GATConv, GCNConv, GCN2Conv, MessagePassing
import torch_geometric.nn.norm as norm
from torch_geometric.data import Data
from src.utils import MLP

class GNNEncoder(nn.Module):
    def __init__(
            self, dim_input_nodes, dim_input_edges, n_layers=5, dim_hidden=64,
            conv_type='SAGE', norm_type='BatchNorm', act='ELU', skip=False, dropout=0.1
    ):
        super(GNNEncoder, self).__init__()
        self.n_layers = n_layers
        self.dim_input_nodes = dim_input_nodes
        self.dim_input_edges = dim_input_edges
        self.dim_hidden = dim_hidden
        self.conv_type = conv_type
        self.norm_type = norm_type
        self.skip = skip

        # initial projection
        self.pre_mlp_nodes = nn.Linear(dim_input_nodes, dim_hidden)
        self.pre_mlp_edges = nn.Linear(dim_input_edges, dim_hidden)

        # Activation function
        self.act = getattr(nn, act)()

        # create convolutions, normalization, and dropout
        conv_li = []
        norm_li_nodes = []
        norm_li_edges = []
        dropout_li_nodes = []
        dropout_li_edges = []
        self.dim_hidden = dim_hidden
        for _ in range(self.n_layers):
            conv_li.append(STR2CONV[conv_type](dim_hidden, dim_hidden))
            if norm_type in ['GraphSizeNorm', 'PairNorm', 'MeanSubtractionNorm']:
                norm_li_nodes.append(getattr(norm, norm_type)())
                norm_li_edges.append(getattr(norm, norm_type)())
            elif norm_type in ['BatchNorm', 'InstanceNorm', 'LayerNorm', 'GraphNorm']:
                norm_li_nodes.append(getattr(norm, norm_type)(dim_hidden))
                norm_li_edges.append(getattr(norm, norm_type)(dim_hidden))
            else:
                assert False, f'unrecognized norm {norm_type}'
            dropout_li_nodes.append(nn.Dropout(p=dropout))
            dropout_li_edges.append(nn.Dropout(p=dropout))
        self.conv_li = nn.ModuleList(conv_li)
        self.norm_li_nodes = nn.ModuleList(norm_li_nodes)
        self.norm_li_edges = nn.ModuleList(norm_li_edges)
        self.dropout_li_nodes = nn.ModuleList(dropout_li_nodes)
        self.dropout_li_edges = nn.ModuleList(dropout_li_edges)

        # skip connection
        if self.skip:
            self.w = nn.Parameter(torch.ones((2, self.n_layers + 1), dtype=torch.float))
        # self.adapters = [None for _ in range(self.n_layers)]

    def add_adapters(self):
        # for param in self.parameters():
        #     param.requires_grad = False
        # adapters = []
        # for _ in range(self.n_layers):
        #     adapters.append(nn.ModuleList([
        #         nn.Sequential(nn.Linear(self.dim_hidden, 32), nn.Linear(32, self.dim_hidden)),
        #         nn.Sequential(nn.Linear(self.dim_hidden, 32), nn.Linear(32, self.dim_hidden))
        #     ]))
        # self.adapters = nn.ModuleList(adapters)
        pass

    def forward(self, emb_nodes, emb_edges, edge_index, graph_node_index=None, graph_edge_index=None):
        V = self.pre_mlp_nodes(emb_nodes)  # (batch_size*N_node_per_graph, hidden_size)
        E = None if emb_edges is None else self.pre_mlp_edges(emb_edges)

        for conv, norm_nodes, norm_edges, dropout_nodes, dropout_edges in \
                zip(
                    self.conv_li, self.norm_li_nodes, self.norm_li_edges,
                    self.dropout_li_nodes, self.dropout_li_edges
                ):
            dV, dE, emb_nodes, emb_edges = \
                conv(
                    V, E, emb_nodes, emb_edges, edge_index,
                    graph_node_index=graph_node_index,
                    graph_edge_index=graph_edge_index
                )
            if self.skip:
                V = V + dV
                E = None if E is None else E + dE
            else:
                V, E = dV, dE

            if self.norm_type == 'BatchNorm':
                V = dropout_nodes(norm_nodes(self.act(V)))
                E = None if E is None else \
                    dropout_edges(norm_edges(self.act(E)))
            else:
                V = dropout_nodes(norm_nodes(self.act(V), batch=graph_node_index))
                E = None if E is None else \
                    dropout_edges(norm_edges(self.act(E), batch=graph_edge_index))
            # if adapter is not None:
            #     V = adapter[0](V) + V
            #     E = adapter[1](E) + E
        # V = V0 + V
        # E = E0 + E
        return V, E


class TIGNN(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super(TIGNN, self).__init__()
        self.phi_e = nn.Sequential(
            nn.Linear(2 * dim_in + 3 + 5, dim_in),
            nn.SiLU(),
            nn.Linear(dim_in, dim_in),
            nn.SiLU(),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(2 * dim_in, dim_in),
            nn.SiLU(),
            nn.Linear(dim_in, dim_out),
        )

    def forward(self, X, E, emb_nodes, emb_edges, edge_index, **kwargs):
        mij = \
            self.phi_e(torch.cat((
                X[edge_index[0]], X[edge_index[1]],
                emb_nodes[edge_index[1]] - emb_nodes[edge_index[0]],
                emb_edges
            ), dim=-1))
        mi = scatter_add(mij, edge_index[1], dim=0, dim_size=X.shape[0])
        X = self.phi_h(torch.cat((X, mi), dim=-1))
        return X, E, emb_nodes, emb_edges


class IGNN(torch.nn.Module):
    def __init__(self, dim_in, dim_out, equivariant=True):
        super(IGNN, self).__init__()
        self.gnn = EGNN(dim_in, dim_out, equivariant=False)

    def forward(self, *args, **kwargs):
        return self.gnn(*args, **kwargs)


class EGNN(torch.nn.Module):
    def __init__(self, dim_in, dim_out, equivariant=True):
        super(EGNN, self).__init__()
        self.equivariant = equivariant
        self.phi_e = nn.Sequential(
            nn.Linear(2 * dim_in + 1 + 2, dim_in),
            nn.SiLU(),
            nn.Linear(dim_in, dim_in),
            nn.SiLU(),
        )
        if equivariant:
            self.phi_x = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.SiLU(),
                nn.Linear(dim_in, 3),
            )
        self.phi_h = nn.Sequential(
            nn.Linear(2 * dim_in, dim_in),
            nn.SiLU(),
            nn.Linear(dim_in, dim_out),
        )

    def forward(self, X, E, emb_nodes, emb_edges, edge_index, graph_node_index=None, **kwargs):
        if graph_node_index is None:
            M = X.shape[0]
        else:
            ones = torch.ones_like(graph_node_index)
            M = torch.zeros(ones.size(0), dtype=ones.dtype, device=X.device)
            M = M.scatter_add(0, graph_node_index, ones)[graph_node_index].unsqueeze(-1)
            # M = torch.scatter_add(torch.ones_like(graph_node_index), graph_node_index)[graph_node_index]

        mij = \
            self.phi_e(torch.cat((
                X[edge_index[0]], X[edge_index[1]],
                torch.norm(emb_nodes[edge_index[1]] - emb_nodes[edge_index[0]], dim=-1).unsqueeze(-1),
                emb_edges
            ), dim=-1))
        if self.equivariant:
            emb_nodes = \
                emb_nodes + 1 / (M - 1) * scatter_add(
                    (emb_nodes[edge_index[1]] - emb_nodes[edge_index[0]]) * self.phi_x(mij),
                    edge_index[1], dim=0, dim_size=emb_nodes.shape[0])
        mi = scatter_add(mij, edge_index[1], dim=0, dim_size=X.shape[0])
        X = self.phi_h(torch.cat((X, mi), dim=-1))

        E = mij

        return X, E, emb_nodes, emb_edges


# class Ours(torch.nn.Module):
#     def __init__(self):
#         pass
#
#     def forward(self, x):
#         # preprocessing
#         coord_plane_intersect = torch.logical_or(C == 0, C == 1).type(torch.float) # [0, 1]^(|V_G| x 3)
#         assert torch.all(C < 100)
#         coord_hash = \
#             torch.cat(
#                 (
#                     100 * coord_plane_intersect + (1-coord_plane_intersect) * C,
#                     graph_index[E[1]]), dim=-1) # [0,...,|V_T|]^|V_G|
#         T = hash(coord_hash)
#
#         geometric = C[E[1]] - C[E[0]]
#         edge_hash = \
#             torch.cat(
#                 (geometric, T[E[1]].unsqueeze(-1), graph_index[E[1]]), dim=-1) # |E_G| x 5
#         D = hash(E_hash) # [0,...,|E_T|]^|E_G|
#
#         G = C[E[1]] - C[E[0]]
#         M = torch.cat((X[E], G), dim=-1)
#         M = scatter_mean(M, D)
#         M = f_msg(M)
#
#         E_1_new = scatter_max(T[E[1]], D)
#         Z = scatter_add(M, E_1_new)
#         Z = f_node(Z)
#         Z = Z[T]
#
#         return Z


class CrystalGNN(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super(CrystalGNN, self).__init__()
        dim_in *= 3
        self.mlp_att = MLP([dim_in, max(dim_in // 4, 16), max(dim_in // 8, 4), 1])
        self.mlp_msg = MLP([dim_in, (dim_in + dim_out) // 2, dim_out])

    def forward(self, X, E, emb_nodes, emb_edges, edge_index, **kwargs):
        H = torch.cat((X[edge_index[0]], X[edge_index[1]], E), dim=-1)
        M = torch.sigmoid(self.mlp_att(H)) * self.mlp_msg(H)
        X = X + scatter_add(src=M, index=edge_index[1], dim=0, dim_size=X.shape[0])
        return X, E, None, None


class GraphNetBlock(MessagePassing):
    """Message passing."""

    def __init__(self, hidden_size, dim_edge_in, dim_node_in):
        super(GraphNetBlock, self).__init__(aggr='add')

        # First net (MLP): eij' = f1(xi, xj, eij)
        self.edge_net = MLP([dim_edge_in, hidden_size],
                            dropout=None, use_norm=False, use_skip=True, act='ELU', gnn_batch=True)

        # Second net (MLP): xi' = f2(xi, sum(eij'))
        self.node_net = MLP([dim_node_in, hidden_size],
                            dropout=None, use_norm=False, use_skip=True, act='ELU', gnn_batch=True)

    def forward(self, X, E, edge_index):
        # Node update
        X_new = self.propagate(edge_index, x=X, edge_attr=E)

        # Edge update
        row, col = edge_index
        E_new = self.edge_net(torch.cat([X[row], X[col], E], dim=-1))

        # Add residuals
        X_new = X_new + X
        E_new = E_new + E

        return X_new, E_new, edge_index

    def message(self, x_i, x_j, edge_attr):
        features = torch.cat([x_i, x_j, edge_attr], dim=-1)

        return self.edge_net(features)

    def update(self, aggr_out, x):
        # aggr_out has shape [num_nodes, out_channels]
        tmp = torch.cat([aggr_out, x], dim=-1)

        return self.node_net(tmp)

    # class GraphNetBlock(MessagePassing):


#     """Message passing."""
#     def __init__(self, hidden_size, dim_edge_in, dim_node_in):
#         super(GraphNetBlock, self).__init__(aggr='mean')

#         # First net (MLP): eij' = f1(xi, xj, eij)
#         self.edge_net = MLP([dim_edge_in, hidden_size, hidden_size, hidden_size],
#                             dropout=None, use_norm=False, use_skip=True, act='ELU', gnn_batch=True)

#         # Second net (MLP): xi' = f2(xi, sum(eij'))
#         self.node_net = MLP([hidden_size, hidden_size, hidden_size, hidden_size],
#                             dropout=None, use_norm=False, use_skip=True, act='ELU', gnn_batch=True)

#     def forward(self, X, E, edge_index):

#         # Node update
#         X_new = self.propagate(edge_index, x= X, edge_attr = E)

#         # # Edge update
#         row, col = edge_index
#         E_new = self.edge_net(torch.cat([X[row], X[col],(X[row] - X[col])**2, E], dim=-1))

#         # # Add residuals
#         X_new = X_new + X
#         E_new = E_new + E
#         # E_new = E

#         return X_new, E_new, edge_index

#     def message(self, x_i, x_j, edge_attr):
#         mij = torch.cat([x_i, x_j, (x_i - x_j)**2, edge_attr], dim=-1)
#         mij = self.edge_net(mij)
#         return (x_i - x_j) * self.node_net(mij)

#     def update(self, aggr_out, x):
#         # aggr_out has shape [num_nodes, out_channels]
#         # tmp = torch.cat([aggr_out, x], dim=-1)
#         x_n = x + aggr_out
#         return x_n

# class MPNN(torch.nn.Module):
#     def __init__(self, latent_size,
#                   name='EncodeProcessDecode'):
#         super(MPNN, self).__init__()

#         # Message passing block
#         self.message_pass = GraphNetBlock(latent_size, latent_size*4, latent_size*4)

#     def forward(self, X, E, emb_nodes, emb_edges, edge_index, **kwargs):
#         X, E, _ = self.message_pass(X, E, edge_index)
#         return X, E, None, None

class MPNN(torch.nn.Module):
    def __init__(self, latent_size,
                 name='EncodeProcessDecode'):
        super(MPNN, self).__init__()

        # Message passing block
        self.message_pass = GraphNetBlock(latent_size, latent_size * 3, latent_size * 2)

    def forward(self, X, E, emb_nodes, emb_edges, edge_index, **kwargs):
        X, E, _ = self.message_pass(X, E, edge_index)
        return X, E, None, None


class GNNWrapper(nn.Module):
    def __init__(self, *args, conv=None, **kwargs):
        super(GNNWrapper, self).__init__()
        assert conv is not None
        dim_in, dim_out, *_ = args
        self.conv = conv(*args)
        self.e_mlp = nn.Linear(dim_in, dim_out)

    def forward(self, X, E, emb_nodes, emb_edges, edge_index, **kwargs):
        if type(self.conv) == GCN2Conv:
            if emb_nodes.shape[-1] == X.shape[-1]:
                x_0 = emb_nodes
            else:
                x_0 = X
            X_new = self.conv(X, x_0, edge_index)
        else:
            X_new = self.conv(X, edge_index)
        if E is not None:
            E = self.e_mlp(E)
        return X_new, E, X, None


def get_gnn_wrapper(conv):
    def f(*args, **kwargs):
        return GNNWrapper(*args, conv=conv, **kwargs)

    return f


STR2CONV = {'TIGNN': TIGNN, 'IGNN': IGNN, 'EGNN': EGNN, 'GCN': get_gnn_wrapper(GCNConv),
            'GCN2': get_gnn_wrapper(GCN2Conv), 'GAT': get_gnn_wrapper(GATConv),
            'crystal': CrystalGNN, 'MPNN': MPNN}