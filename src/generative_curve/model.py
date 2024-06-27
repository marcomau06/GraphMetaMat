from src.layers_encoder import GNNEncoder
from src.layers_pooler import Pooler
import src.layers_decoder_shape as lds
import src.layers_decoder_magnitude as ldm

from src.dataset_preprocessing_collated import RESOLUTION_CURVE

import numpy as np
import random
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from src.config import args, ETH_FULL_C_VECTOR


'''
norm_stress_strain
split_stress_strain
merge_stress_strain
unnorm_stress_strain
'''


# class ModelFactory:
#     def __init__(self, encoder_config, decoder_config, curve_resolution, loss_name, apply_log, use_accum=False):
#         self.encoder_config = encoder_config
#         self.decoder_config = decoder_config
#         self.curve_resolution = curve_resolution
#         self.loss_name = loss_name
#         self.apply_log = apply_log
#         self.use_accum = use_accum
#
#     def get_model(self, dim_input_nodes, dim_input_edges):
#         encoder = GNNEncoder(dim_input_nodes, dim_input_edges, **self.encoder_config)
#         decoder = \
#             PoolAutoregressiveDecoder(
#                 dim_in=self.encoder_config['dim_hidden'],
#                 curve_resolution=self.curve_resolution,
#                 **self.decoder_config
#             )
#         model = Model(encoder, decoder, self.loss_name, self.apply_log, self.use_accum)
#         return model

class ModelEnsemble(nn.Module):
    def __init__(self, model_li, ensemble_size_max=5):
        super().__init__()
        self.model_li = nn.ModuleList(model_li[-ensemble_size_max:])
        self.use_snapshot = True

    def inference(self, *args, **kwargs):
        c_shape_li, c_magnitude_li, cn_li = [], [], []
        for model in self.model_li:
            if 'return_cn' in kwargs and kwargs['return_cn']:
                c_shape, c_magnitude, *_, cn = model.inference(*args, **kwargs)
                cn_li.append(cn)
            else:
                c_shape, c_magnitude, *_ = model.inference(*args, **kwargs)
            c_shape_li.append(c_shape)
            c_magnitude_li.append(c_magnitude)
        c_shape, c_shape_uncertainty = self.merge_output_li(c_shape_li)
        c_magnitude, c_magnitude_uncertainty = self.merge_output_li(c_magnitude_li)
        if 'return_cn' in kwargs and kwargs['return_cn']:
            cn_mean, _ = self.merge_output_li(cn_li)
            return c_shape, c_magnitude, c_shape_uncertainty, c_magnitude_uncertainty, cn_mean
        else:
            return c_shape, c_magnitude, c_shape_uncertainty, c_magnitude_uncertainty

    def get_emb(self, *args, **kwargs):
        emb_nodes_out, emb_edges_out = [], []
        for model in self.model_li:
            emb_nodes, emb_edges = model.get_emb(*args, **kwargs)
            emb_nodes_out.append(emb_nodes)
            emb_edges_out.append(emb_edges)
        emb_nodes_out = torch.mean(torch.stack(emb_nodes_out, dim=0), dim=0)
        emb_edges_out = torch.mean(torch.stack(emb_edges_out, dim=0), dim=0)
        return emb_nodes_out, emb_edges_out

    def split_stress_strain(self, *args, **kwargs):
        return self.model_li[-1].split_stress_strain(*args, **kwargs)

    def norm_stress_strain(self, *args, **kwargs):
        return self.model_li[-1].norm_stress_strain(*args, **kwargs)

    def merge_output_li(self, output_li):
        output = torch.mean(torch.stack(output_li, dim=0), dim=0)
        output_uncertainty = torch.std(torch.stack(output_li, dim=0), dim=0)
        return output, output_uncertainty

    @classmethod
    def from_path(cls, dn, model_template):
        # dn = os.path.split(pn)[0]
        model_li = []
        snapshot_model_fn_li = [x for x in os.listdir(dn) if ('model_epoch_snapshot' in x and '.pt' in x)]
        snapshot_model_fn_li = sorted(snapshot_model_fn_li, key=lambda x: int(x.split('_')[-1].split('.pt')[0]))#, reverse=True)
        n_snapshots = int(args['forward']['train_config']['use_snapshot']['num_snapshots'])
        for fn in snapshot_model_fn_li[:n_snapshots]:
            if '.pt' in fn:
                pn = os.path.join(dn, fn)
                model = copy.deepcopy(model_template)
                model.encoder.add_adapters()
                model.load_state_dict(torch.load(pn, map_location=torch.device(args['device'])))
                model_li.append(model)
        return cls(model_li)


class Model(nn.Module):
    def __init__(self, encoder, encoder_rho, pooler, decoder_shape, decoder_magnitude, loss_coeff,
                 data_loader, loss_name='mse'):
        super(Model, self).__init__()
        self.encoder = encoder
        self.encoder_rho = encoder_rho
        self.pooler = pooler
        self.decoder_shape = decoder_shape
        self.decoder_magnitude = decoder_magnitude
        self.c_stats = data_loader.dataset.c_stats
        self.loss_name = loss_name
        self.loss_coeff = loss_coeff

    @classmethod
    def init_from_cfg(cls, dataset, dim, encoder_cfg, pooler_cfg, decoder_shape_nm, decoder_shape_cfg,
                      decoder_magnitude_nm, decoder_magitude_cfg, loss_name, loss_coeff):
        dim_input_nodes = dataset.dataset.dim_input_nodes
        dim_input_edges = dataset.dataset.dim_input_edges
        dim_curve = dataset.dataset.dim_curve

        encoder = \
            GNNEncoder(
                dim_input_nodes=dim_input_nodes,
                dim_input_edges=dim_input_edges,
                dim_hidden=dim, **encoder_cfg)
        encoder_rho = nn.Sequential(
            nn.Linear(1, dim), nn.ELU(), nn.Linear(dim, dim)
        )
        #RhoEncoder(dim=dim, min=0.02, max=0.3)
        pooler = Pooler(dim, pool_name=pooler_cfg['pool_name'])
        curve_resolution = RESOLUTION_CURVE # if dataset.dataset.digitize_cfg is None else dataset.dataset.digitize_cfg['n_freq']
        decoder_shape = \
            getattr(lds, decoder_shape_nm)(
                dim_in=dim, dim_out=dim_curve,
                curve_resolution=curve_resolution,
                merge_name=pooler_cfg['merge_name'],
                **decoder_shape_cfg)
        if not ETH_FULL_C_VECTOR:
            assert dim_curve == 1 # need to be 1 if we use magn = C*rho^n
        decoder_magnitude = \
            getattr(ldm, decoder_magnitude_nm)(
                dim_in=dim, dim_out=dim_curve if ETH_FULL_C_VECTOR else 2*dim_curve,
                c_stats=dataset.dataset.c_stats,
                merge_name=pooler_cfg['merge_name'],
                **decoder_magitude_cfg)
        return cls(
            encoder, encoder_rho, pooler, decoder_shape, decoder_magnitude, loss_coeff,
            data_loader=dataset, loss_name=loss_name)

    def get_emb(self, graph):
        emb_nodes, emb_edges = \
            self.encoder(
                graph.feats_node, graph.feats_edge, graph.edge_index,
                graph_node_index=graph.graph_node_index,
                graph_edge_index=graph.graph_edge_index
            )
        emb_nodes, emb_edges = self.pooler(emb_nodes, emb_edges, graph)
        return emb_nodes, emb_edges

    def inference(self, graph, curve=None, cur_epoch=None, return_cn=False):
        emb_nodes, emb_edges = \
            self.encoder(
                graph.feats_node, graph.feats_edge, graph.edge_index,
                graph_node_index=graph.graph_node_index,
                graph_edge_index=graph.graph_edge_index
            )
        emb_rho = self.encoder_rho(graph.feats_rho.unsqueeze(-1))
        emb_nodes, emb_edges = self.pooler(emb_nodes, emb_edges, graph)
        c_shape = \
            self.decoder_shape(
                emb_nodes=emb_nodes,
                emb_edges=emb_edges,
                emb_rho=emb_rho,
                curve=curve
            )
        c_magnitude = \
            self.decoder_magnitude(
                emb_nodes=emb_nodes,
                emb_edges=emb_edges,
                emb_rho=emb_rho,
                rho=graph.feats_rho,
                return_cn=return_cn
            )
        if return_cn:
            c_magnitude, cn = c_magnitude
            out = c_shape, c_magnitude, None, None, cn
        else:
            out = c_shape, c_magnitude, None, None
        assert not torch.isnan(c_magnitude).any()
        return out

    def unfreeze_encoder(self):
        pass

    def forward(self, graph, curve, graph_pos=None, graph_neg=None, cur_epoch=None):
        '''
        curve_original, curve_stats = curve.curve, curve.curve_stats
        strain, stress_shape, stress_magnitude = self.split_stress_strain(curve_true) # disentangle shape and magnitude
        strain, stress_magnitude = self.norm_stress_strain(strain, stress_magnitude, curve_stats) # divide by stats
        stress_shape_delta = self.stress2delta_stress(stress_shape) # compute delta stress
        stress_shape = self.delta_stress2stress(stress_shape_delta)
        strain, stress_magnitude = self.unnorm_stress_strain(strain, stress_magnitude, curve_stats)
        curve = self.merge_stress_strain(strain, stress_shape, stress_magnitude)
        '''
        c_shape, c_magnitude, *_ = self.inference(graph, curve=curve, cur_epoch=cur_epoch)

        # loss function
        if self.loss_name == 'mse':
            loss_fn = F.mse_loss
        elif self.loss_name == 'huber':
            loss_fn = F.huber_loss
        else:
            assert False

        loss_shape = None if ETH_FULL_C_VECTOR else loss_fn(c_shape, curve.c_shape)#, reduce=False)
        loss_magnitude = loss_fn(c_magnitude, curve.c_magnitude)
        loss_smoothness = torch.mean((c_shape[:, :-1] - c_shape[:, 1:]) ** 2)

        if ETH_FULL_C_VECTOR:
            loss = loss_magnitude
        else:
            # self.loss_coeff['main_coeff'] * loss_main + \
            loss = \
                self.loss_coeff['magnitude_coeff'] * loss_magnitude + \
                self.loss_coeff['shape_coeff'] * loss_shape + \
                self.loss_coeff['smoothness_coeff'] * loss_smoothness

        c_magnitude_mean, c_magnitude_std, *_ = self.c_stats
        # c_magnitude_max = (torch.log10(2.7*graph.feats_rho[0].cpu()) - c_magnitude_mean)/c_magnitude_std
        loss = loss # + 100.0 * torch.mean(torch.clamp(c_magnitude.view(-1)-c_magnitude_max.to(c_magnitude), min=0.0))

        loss_monotonic = torch.tensor(0.0, device=curve.c_shape.device)
        NUM_AUGMENTATIONS = 0
        for _ in range(NUM_AUGMENTATIONS):
            rho_delta = 0.02 * torch.randn_like(graph.feats_rho)
            graph.feats_rho = graph.feats_rho + rho_delta
            _, c_magnitude_aug, *_ = self.inference(graph, curve=curve)
            loss_monotonic = \
                loss_monotonic + self.aug_loss(
                    curve.c_magnitude,
                    c_magnitude_aug,
                    2 * ((rho_delta >= 0.0).float() - 0.5).to(args['device'])
                ) / NUM_AUGMENTATIONS
            # loss_monotonic = loss_monotonic + \
            #     torch.mean(torch.maximum(
            #         2 * ((rho_delta >= 0.0).float() - 0.5).to(args['device']) * \
            #         (curve.c_magnitude - c_magnitude_aug).view(-1),
            #         torch.zeros_like(c_magnitude).view(-1))**2
            #     ) / NUM_AUGMENTATIONS
            graph.feats_rho = graph.feats_rho - rho_delta

        NUM_AUGMENTATIONS = 0
        for _ in range(NUM_AUGMENTATIONS):
            c_shape_aug_rho, mask_rho = self.augment_rho(graph, curve)
            c_shape_aug_over, mask_over = self.augment_overhang(graph, curve)
            # thresh = get_thresh(c_shape_aug_rho.shape[1])
            s, e = get_thresh(c_shape_aug_rho.shape[1], 4000), get_thresh(c_shape_aug_rho.shape[1], 12000)
            # mask_thresh = curve.c_shape * self.c_stats[3].to(curve.c_shape.device) + self.c_stats[2].to(curve.c_shape.device) < 5
            loss_aug = \
                self.aug_loss(
                    curve.c_shape[:, s:e],#[mask_thresh],#[:, :thresh],
                    c_shape_aug_rho[:, s:e],#[mask_thresh],#[:, :thresh],
                    # get_second_deriv(curve.c_shape[:, :thresh], dim=1),
                    # get_second_deriv(c_shape_aug_rho[:, :thresh], dim=1),
                    mask_rho
                ) / NUM_AUGMENTATIONS
            loss_monotonic = loss_monotonic + loss_aug
            loss_aug = \
                self.aug_loss(
                    curve.c_shape[:, s:e],#[mask_thresh],#[:, :thresh],
                    c_shape_aug_over[:, s:e],#[mask_thresh],#[:, :thresh],
                    # get_second_deriv(curve.c_shape[:, :thresh], dim=1),
                    # get_second_deriv(c_shape_aug_over[:, :thresh], dim=1),
                    mask_over
                ) / NUM_AUGMENTATIONS
            loss_monotonic = loss_monotonic + loss_aug


            # rho_delta = torch.randn_like(graph.feats_rho)
            # graph.feats_rho = graph.feats_rho + rho_delta
            # c_shape_aug, _, *_ = self.inference(graph, curve=curve)
            # loss_monotonic = loss_monotonic + \
            #     torch.mean(torch.maximum(
            #         (2 * ((rho_delta >= 0.0).float() - 0.5).to(args['device'])).unsqueeze(-1).unsqueeze(-1) * \
            #         (curve.c_shape - c_shape_aug),
            #         torch.zeros_like(c_shape_aug))**2
            #     ) / NUM_AUGMENTATIONS
            # graph.feats_rho = graph.feats_rho - rho_delta
        loss = loss + loss_monotonic

        if graph_pos is not None:
            assert graph_neg is not None
            loss_contrastive = self.contrast_loss(graph, graph_pos, graph_neg)
            loss = loss + loss_contrastive

        # second_deriv_pred = c_shape[:,2:] + c_shape[:,:-2] - 2.0 * c_shape[:,1:-1]
        # second_deriv_true = curve.c_shape[:,2:] + curve.c_shape[:,:-2] - 2.0 * curve.c_shape[:,1:-1]
        # loss = loss + 5.0 * F.mse_loss(second_deriv_pred, second_deriv_true)
        return loss

    def contrast_loss(self, graph, graph_pos, graph_neg):
        emb_graph = self.get_contrast_emb(graph)
        emb_graph_pos = self.get_contrast_emb(graph_pos)
        emb_graph_neg = self.get_contrast_emb(graph_neg)
        logits_pos = torch.einsum('ij, ij -> i', emb_graph, emb_graph_pos)
        logits_neg = torch.einsum('ij, ij -> i', emb_graph, emb_graph_neg)
        loss = F.cross_entropy(
            torch.stack((logits_pos, logits_neg), dim=-1),
            torch.zeros_like(logits_pos, dtype=torch.long, device=logits_pos.device)
        )
        return loss

    def get_contrast_emb(self, graph):
        emb_nodes, emb_edges = \
            self.encoder(
                graph.feats_node, graph.feats_edge, graph.edge_index,
                graph_node_index=graph.graph_node_index,
                graph_edge_index=graph.graph_edge_index
            )
        emb_nodes, emb_edges = self.pooler(emb_nodes, emb_edges, graph)
        emb_rho = self.encoder_rho(graph.feats_rho.unsqueeze(-1))
        x = self.decoder_shape.merger(emb_nodes, emb_edges, emb_rho)
        return x

### AUGMENTATION FUNCTIONS ###
    def augment_rho(self, graph, curve):
        rho_delta = torch.randn_like(graph.feats_rho)
        graph.feats_rho = graph.feats_rho + rho_delta
        c_shape_aug, _, *_ = self.inference(graph, curve=curve)
        graph.feats_rho = graph.feats_rho - rho_delta
        mask = (2 * ((rho_delta >= 0.0).float() - 0.5).to(args['device'])).unsqueeze(-1).unsqueeze(-1)
        return c_shape_aug, mask

    def augment_overhang(self, graph, curve):
        aid = int(random.random()*10)
        pn = '/home/derek/Documents/tmp'
        if not os.path.isdir(pn):
            os.mkdir(pn)

        bid = "_".join([str(g.graph["gid"]) for g in graph.g_li])
        pn_cache = os.path.join(pn, f'{bid}_-_{aid}.pkl')
        if os.path.exists(pn_cache):
            graph_aug = None
            c_shape_aug, _, *_ = self.inference(graph_aug, curve=curve)
        else:
            edge_index_original = deepcopy(graph.edge_index)
            feats_edge_original = deepcopy(graph.feats_edge)
            graph_edge_index_original = deepcopy(graph.graph_edge_index)
            dim_size = graph.edge_index.max()+1
            count0 = get_count(graph.edge_index[0], dim_size=dim_size)
            count1 = get_count(graph.edge_index[1], dim_size=dim_size)

            edge_index_mask = torch.logical_and(torch.gt(count0, 1), torch.gt(count1, 1))
            for i in range(dim_size):
                edge_index_mask[torch.nonzero(graph.edge_index[0] == i)[0]] = False
            p_to_mask = random.random()*0.3

            edge_index_mask = \
                torch.logical_and(
                    edge_index_mask, torch.lt(
                        torch.rand(edge_index_mask.shape, device=edge_index_mask.device), p_to_mask
                    ))
            graph.edge_index = graph.edge_index[:,torch.logical_not(edge_index_mask)]
            graph.feats_edge = graph.feats_edge[torch.logical_not(edge_index_mask)]
            graph.graph_edge_index = graph.graph_edge_index[torch.logical_not(edge_index_mask)]

            c_shape_aug, _, *_ = self.inference(graph, curve=curve)

            graph.edge_index = edge_index_original
            graph.feats_edge = feats_edge_original
            graph.graph_edge_index = graph_edge_index_original
        return c_shape_aug, None

    def aug_loss(self, curve_c_shape, c_shape_aug, mask=None):
        if mask is None:
            mask = -torch.ones_like(curve_c_shape, device=curve_c_shape.device)
        # loss_augmented = \
        #      torch.mean(torch.maximum(
        #          mask * (curve_c_shape - c_shape_aug),
        #          torch.zeros_like(c_shape_aug)
        #      ) ** 2)
        loss_augmented = \
             torch.mean(torch.maximum(
                 mask * (curve_c_shape - c_shape_aug),
                 torch.zeros_like(c_shape_aug)
             ))
        return loss_augmented

from torch_scatter import scatter_add
def get_count(arr, dim_size):
    assert arr.max() < dim_size
    out = scatter_add(torch.ones_like(arr), arr, dim_size=dim_size)
    return out[arr]

def get_second_deriv(arr, dim):
    arr = arr.swapaxes(dim, 0)
    arr = 2*arr[1:-1] - arr[:-2] - arr[2:]
    arr = arr.swapaxes(dim, 0)
    return arr

def get_thresh(len_arr, x=4000):
    # return int((4000-1000)/(12000-1000)*len_arr)
    return int((x-1000)/(12000-1000)*len_arr)
