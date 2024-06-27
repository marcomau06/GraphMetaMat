import math
import torch
import torch.nn.functional as F

import numpy as np
from src.config import args

class RAGDecoder(torch.nn.Module):
    def __init__(self, dim_in, data_loader, encoder, pooler):
        super(RAGDecoder, self).__init__()
        keys = []
        values_shape = []
        values_magni = []
        with torch.no_grad():
            for graph, curve in data_loader:
                emb_nodes, emb_edges = \
                    encoder(
                        graph.feats_node, graph.feats_edge, graph.edge_index,
                        graph_node_index=graph.graph_node_index,
                        graph_edge_index=graph.graph_edge_index
                    )
                emb_graphs = pooler(emb_nodes, emb_edges, graph)
                keys.append(emb_graphs)
                if curve.c_magnitude is not None:
                    values_magni.append(curve.c_magnitude.cpu().numpy())
                if curve.c_shape is not None:
                    values_shape.append(curve.c_shape.cpu().numpy())

        self.keys, self.keys_l2_norm = self._construct_keys(keys)
        self.values_magni, self.values_shape = \
            self._construct_values(values_magni, values_shape)
        self.kernel = Bilinear(dim_in)
        self.k = 5

    def reset_datastore(self, data_loader, encoder, pooler):
        keys = []
        values_shape = []
        values_magni = []
        with torch.no_grad():
            for graph, curve in data_loader:
                graph.to(args['device'])
                curve.to(args['device'])
                emb_nodes, emb_edges = \
                    encoder(
                        graph.feats_node, graph.feats_edge, graph.edge_index,
                        graph_node_index=graph.graph_node_index,
                        graph_edge_index=graph.graph_edge_index
                    )
                emb_graphs = pooler(emb_nodes, emb_edges, graph)
                keys.append(emb_graphs)
                if curve.c_magnitude is not None:
                    values_magni.append(curve.c_magnitude.cpu().numpy())
                if curve.c_shape is not None:
                    values_shape.append(curve.c_shape.cpu().numpy())

        self.keys, self.keys_l2_norm = self._construct_keys(keys)
        self.values_magni, self.values_shape = \
            self._construct_values(values_magni, values_shape)


    def _construct_keys(self, keys):
        keys = torch.cat(keys, dim=0)
        keys_l2_norm = torch.norm(keys, dim=1).unsqueeze(-1)
        return keys/keys_l2_norm, keys_l2_norm

    def _construct_values(self, values_magni, values_shape):
        if len(values_magni) > 0:
            values_magni = np.concatenate(values_magni, axis=0)
        else:
            values_magni = None

        if len(values_shape) > 0:
            values_shape = np.concatenate(values_shape, axis=0)
        else:
            values_shape = None

        return values_magni, values_shape

    def forward(self, emb_graphs, our_shape, our_magnitude):
        self.keys = self.keys.to(emb_graphs.device)
        self.keys_l2_norm = self.keys_l2_norm.to(emb_graphs.device)
        if True:#self.training:
            out_magni, out_shape = our_magnitude, our_shape
        else:
            kernel_weights = \
                torch.matmul(
                    emb_graphs/torch.norm(emb_graphs, dim=1).unsqueeze(-1),
                    self.keys.transpose(0,1))  # N x k
            _, topk_indices = torch.topk(kernel_weights, k=self.k)

            kernel_weights = \
                kernel_weights[
                    torch.arange(
                        topk_indices.shape[0], device=topk_indices.device
                    ).unsqueeze(1).repeat(1, topk_indices.shape[1]),
                    topk_indices]
            # kernel_weights = \
            #     torch.cat((
            #         kernel_weights[
            #             torch.arange(
            #                 topk_indices.shape[0], device=topk_indices.device
            #             ).unsqueeze(1).repeat(1,topk_indices.shape[1]),
            #             topk_indices],
            #         torch.ones((kernel_weights.shape[0],1), device=kernel_weights.device)
            #     ), dim=1) # N x (k+1)
            kernel_weights = F.softmax(kernel_weights, dim=1)

            if self.values_magni is None:
                assert our_magnitude is None
                out_magni = None
            else:
                values_magni = torch.tensor(self.values_magni[topk_indices.cpu()], device=emb_graphs.device) # N x k
                out_magni = \
                    torch.sum(kernel_weights.unsqueeze(-1) * values_magni, dim=1)
                # out_magni = \
                #     torch.sum(
                #         kernel_weights.unsqueeze(-1) * torch.cat((
                #             values_magni, our_magnitude.unsqueeze(1).detach()), dim=1), dim=1)
                out_magni = our_magnitude - our_magnitude.detach() + out_magni

            if self.values_shape is None:
                assert our_shape is None
                out_shape = None
            else:
                values_shape = torch.tensor(self.values_shape[topk_indices.cpu()], device=emb_graphs.device) # N x k
                out_shape = \
                    torch.sum(kernel_weights.unsqueeze(-1).unsqueeze(-1) * values_shape, dim=1)
                # values_shape = torch.tensor(self.values_shape[topk_indices.cpu()], device=emb_graphs.device) # N x (k+1) x L
                # out_shape = \
                #     torch.sum(
                #         kernel_weights.unsqueeze(-1).unsqueeze(-1) * torch.cat((
                #             values_shape, our_shape.unsqueeze(1).detach()), dim=1), dim=1)
                out_shape = our_shape - our_shape.detach() + out_shape

        return out_magni, out_shape

# def forward(self, emb_graphs, our_shape, our_magnitude):
#     self.keys = self.keys.to(emb_graphs.device)
#     self.keys_l2_norm = self.keys_l2_norm.to(emb_graphs.device)
#     if self.training:
#         mask = torch.ones(self.keys.shape[0], dtype=torch.bool, device=emb_graphs.device)
#         mask[np.random.choice(self.keys.shape[0], int(self.keys.shape[0]/10), replace=False)] = False
#         _, topk_indices = torch.topk(torch.matmul(
#             emb_graphs/torch.norm(emb_graphs, dim=1).unsqueeze(-1),
#             self.keys[mask].transpose(0,1)), k=self.k+1)
#         topk_indices = topk_indices[:, 1:]
#     else:
#         _, topk_indices = torch.topk(torch.matmul(
#             emb_graphs/torch.norm(emb_graphs, dim=1).unsqueeze(-1),
#             self.keys.transpose(0,1)), k=self.k)
#
#     retrieved_keys = self.keys[topk_indices]*self.keys_l2_norm[topk_indices]
#     kernel_weights = self.kernel(emb_graphs, retrieved_keys) # N x k
#     kernel_weights = \
#         torch.cat((
#             kernel_weights,
#             torch.ones((kernel_weights.shape[0],1), device=kernel_weights.device)
#         ), dim=1) # N x (k+1)
#     kernel_weights = F.softmax(kernel_weights, dim=1)
#
#     if self.values_magni is None:
#         assert our_magnitude is None
#         out_magni = None
#     else:
#         values_magni = torch.tensor(self.values_magni[topk_indices.cpu()], device=emb_graphs.device) # N x k
#         out_magni = \
#             torch.sum(
#                 kernel_weights.unsqueeze(-1) * torch.cat((
#                     values_magni, our_magnitude.unsqueeze(1).detach()), dim=1), dim=1)
#         out_magni = our_magnitude - our_magnitude.detach() + out_magni
#
#     if self.values_shape is None:
#         assert our_shape is None
#         out_shape = None
#     else:
#         values_shape = torch.tensor(self.values_shape[topk_indices.cpu()], device=emb_graphs.device) # N x (k+1) x L
#         out_shape = \
#             torch.sum(
#                 kernel_weights.unsqueeze(-1).unsqueeze(-1) * torch.cat((
#                     values_shape, our_shape.unsqueeze(1).detach()), dim=1), dim=1)
#         out_shape = our_shape - our_shape.detach() + out_shape
#
#     return out_magni, out_shape

class Bilinear(torch.nn.Module):
    def __init__(self, dim_in):
        super(Bilinear, self).__init__()
        bound = 1 / math.sqrt(dim_in)
        self.bilin_mat = torch.nn.Parameter(2*bound*(torch.rand((dim_in, dim_in))-0.5))
        self.bias = torch.nn.Parameter(2*bound*(torch.rand(1)-0.5))

    def forward(self, in1, in2):
        out = torch.einsum('bi, ij, bkj -> bk', in1, self.bilin_mat, in2) + self.bias
        return out
