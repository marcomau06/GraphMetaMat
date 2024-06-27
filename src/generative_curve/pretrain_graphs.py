# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:46:55 2023

@author: Marco Maurizi
"""

import copy

import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import torch.nn.functional as F
import torch
import os
import random

from src.generative_curve.pretrain import NTXentLoss, write_model
from src.dataset import GraphObj, GraphObjCollated
from src.utils import log_dir

def perturb_add_edges(graph):
    assert len(graph.nodes()) > 3
    perturb_id = random.choice([0,1,2])
    existing_edges = set([tuple(sorted(list(eid))) for eid in graph.edges()])
    all_edges = set([tuple(sorted([nid_u, nid_v])) for nid_u in graph.nodes() for nid_v in graph.nodes()])
    assert len(all_edges - existing_edges) > 2
    if perturb_id == 0:
        sampled_edge_li = None
    elif perturb_id == 1:
        degree_min = min(dict(graph.degree()).values())
        valid_nodes = \
            [nid for nid in graph.nodes() if graph.degree(nid) > degree_min]
        valid_edges = \
            {tuple(sorted([nid_u, nid_v])) for nid_u in valid_nodes for nid_v in valid_nodes if nid_u != nid_v}
        if len(valid_edges - valid_edges) > 1:
            sampled_edge = random.choice(list(valid_edges - existing_edges))
        else:
            sampled_edge = random.choice(list(all_edges - existing_edges))
        sampled_edge_li = [sampled_edge]
    elif perturb_id == 2:
        degree_min = min(dict(graph.degree()).values())
        valid_nodes = \
            [nid for nid in graph.nodes() if graph.degree(nid) > degree_min]
        valid_edges = \
            {tuple(sorted([nid_u, nid_v])) for nid_u in valid_nodes for nid_v in valid_nodes if nid_u != nid_v}
        if len(valid_edges - valid_edges) > 1:
            sampled_edge_stage_1 = random.choice(list(valid_edges - existing_edges))
        else:
            sampled_edge_stage_1 = random.choice(list(all_edges - existing_edges))
        if len(valid_edges - valid_edges) > 1:
            sampled_edge_stage_2 = random.choice(list(valid_edges - existing_edges - {sampled_edge_stage_1}))
        else:
            sampled_edge_stage_2 = random.choice(list(all_edges - existing_edges - {sampled_edge_stage_1}))
        sampled_edge_li = [sampled_edge_stage_1, sampled_edge_stage_2]
    else:
        assert False
    if sampled_edge_li is not None:
        edge_feats = dict(graph.edges[random.choice(list(graph.edges()))])
        graph.add_edges_from(sampled_edge_li, **edge_feats)
    return graph

def perturb_rm_edges(graph):
    assert len(graph.edges()) > 3
    perturb_id = random.choice([0,1,2])
    if perturb_id == 0:
        sampled_edge_li = None
    elif perturb_id == 1:
        degree_min = min(dict(graph.degree()).values())
        valid_edges = \
            [eid for eid in graph.edges() 
             if min(graph.degree(eid[0]), graph.degree(eid[1])) > degree_min]
        sampled_edge = \
            random.choice(valid_edges) \
                if len(valid_edges) > 1 else \
            random.choice(list(graph.edges()))
        sampled_edge_li = [sampled_edge]
    elif perturb_id == 2:
        degree_min = min(dict(graph.degree()).values())
        valid_edges_stage_1 = \
            [eid for eid in graph.edges() 
             if min(graph.degree(eid[0]), graph.degree(eid[1])) > degree_min]
        sampled_edge_stage_1 = \
            random.choice(valid_edges_stage_1) \
                if len(valid_edges_stage_1) > 1 else \
            random.choice(list(graph.edges()))
        valid_edges_stage_2 = \
            [eid for eid in graph.edges() 
             if min(graph.degree(eid[0]), graph.degree(eid[1])) > degree_min+1]
        sampled_edge_stage_2 = \
            random.choice(list(set(valid_edges_stage_2) - {sampled_edge_stage_1})) \
                if len(set(valid_edges_stage_2) - {sampled_edge_stage_1}) > 1 else \
            random.choice(list(set(graph.edges()) - {sampled_edge_stage_1}))
        sampled_edge_li = [sampled_edge_stage_1, sampled_edge_stage_2]
    else:
        assert False
    if sampled_edge_li is not None:
        graph.remove_edges_from(sampled_edge_li)    
    return graph

def perturb_radiuses(graph):
    perturb_id = random.choice([0,1,2])
    if perturb_id == 0:
        pass
    elif perturb_id == 1:
        perturb_edges = random.choices(list(graph.edges()), k=int(len(graph.edges())/2))
        for eid in perturb_edges:
            graph.edges[eid]['radius'] = 0.5*graph.edges[eid]['radius']
    elif perturb_id == 2:
        perturb_edges = random.choices(list(graph.edges()), k=int(len(graph.edges())/2))
        for eid in perturb_edges:
            graph.edges[eid]['radius'] = 2.0*graph.edges[eid]['radius']
    else:
        assert False
    return graph

def perturb_coords(graph):
    perturb_id = random.choice([0,1,2])
    if perturb_id == 0:
        perturb_nodes = None
    elif perturb_id == 1:
        perturb_nodes = random.choices(list(graph.nodes()), k=1)
    elif perturb_id == 2:
        perturb_nodes = random.choices(list(graph.nodes()), k=2)
    else:
        assert False
    if perturb_nodes is not None:
        node_feats = dict(graph.nodes[random.choice(list(set(graph.nodes()) - set(perturb_nodes)))])
        for nid in perturb_nodes:
            for k, v in node_feats.items():
                graph.nodes[nid][k] = v
    return graph

def invariant_perturb(graph_batch, node_feat_cfg, edge_feat_cfg):
    g_li = copy.deepcopy(graph_batch.g_li)
    for g in g_li:
        g = perturb_add_edges(g)
        g = perturb_rm_edges(g)
        g = perturb_coords(g)
        g = perturb_radiuses(g)
    graph_batch = GraphObjCollated.collate([GraphObj(g, node_feat_cfg, edge_feat_cfg) for g in g_li])
    return graph_batch

# def cvt_graph(g, feat_cfg):
#     graph_collated, node_feat_cfg, edge_feat_cfg = feat_cfg
#     assert len(graph_collated.g_li) == 1
#     graph_feat_cfg = graph_collated.g_li[0].graph
#     for nid in g.nodes():
#         g.nodes[nid]['coord'] = (g.nodes[nid]['coord'] + 1) / 2
#         if 'Es' in graph_feat_cfg:
#             g.nodes[nid]['Es'] = graph_feat_cfg['Es']
#     for eid in g.edges():
#         nid_start, nid_end = eid
#         g.edges[eid]['length'] = \
#             np.linalg.norm(g.nodes[nid_start]['coord'] - g.nodes[nid_end]['coord'])
#         # g.edges[eid]['radius'] = r # already predicted and included as edge feature
#     g.graph = graph_feat_cfg
#     if type(node_feat_cfg) is not list:
#         node_feat_cfg = node_feat_cfg.split(',')
#     if type(edge_feat_cfg) is not list:
#         edge_feat_cfg = edge_feat_cfg.split(',')
#     graph = GraphObj(g, node_feat_cfg, edge_feat_cfg)
#     return graph

def pretrain_graphs(dataset, dataset_pretrain, model, optimizer, train_cfg, node_feat_cfg, edge_feat_cfg, device, **kwargs):
    model.train()
    model = model.to(device)

    best_iter = -1
    num_iters = 0
    num_epochs = 0
    loss_fn = NTXentLoss(device)
    loss_pt = []
    while True:
        for graph1 in dataset_pretrain:
            optimizer.zero_grad()

            # get the representations and the projections
            graph1.to(device)
            graph2 = invariant_perturb(graph1, node_feat_cfg, edge_feat_cfg)
            graph2.to(device)
          
            zis = model.get_emb_all(graph1)  # [N,C]
            zjs = model.get_emb_all(graph2)  # [N,C]            
            
            # normalize projection feature vectors
            zis = F.normalize(zis, dim=1)
            zjs = F.normalize(zjs, dim=1)
            
            # get loss
            labels = torch.tensor([min(dict(g.degree()).values()) for g in graph1.g_li])
            loss = loss_fn(zis, zjs, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad)
            optimizer.step()
            
            # Store training loss values
            loss_pt.append(float(loss.detach().cpu()))
            
            if num_iters % 10 == 0:
                print(f'loss [{num_iters}/{train_cfg.max_update} iters]: {float(loss.detach().cpu())}')
                # writer.add_scalar('train/loss', loss.item(), num_iters)
            if num_iters % train_cfg.save_interval == 0:
                pretrain_test(model, dataset, dataset_pretrain, num_iters, device)
                write_model(model, log_dir, num_iters)
            if num_iters >= train_cfg.max_update:
                write_model(model, log_dir, 'pretrained')
                return best_iter, loss_pt
            num_iters += 1
        if not train_cfg.no_epoch_checkpoints:
            write_model(model, log_dir, f'epoch_{num_epochs}')
        num_epochs += 1
        print(f'iterated through whole dataset {num_epochs} times')

def pretrain_test(model, dataset, dataset_pretrain, cur_iter, device):
    zi_li = []
    labels_li = []   
    # Pretraining dataset (unlabelled)
    for graph in dataset_pretrain:
        graph.to(device)
        zis = model.get_emb_all(graph) # [N,C]
        zis = F.normalize(zis, dim=1).detach().cpu().numpy()
        zi_li.append(zis)
        labels_li.append(np.array([1]*zis.shape[0]))
        
    # Supervised training dataset (labelled)
    magnitude = []
    zi_sv = []
    for graph, curve in dataset:
        graph.to(device)
        zis = model.get_emb_all(graph) # [N,C]
        zis = F.normalize(zis, dim=1).detach().cpu().numpy()       
        zi_li.append(zis)
        labels_li.append(np.array([0]*zis.shape[0]))
        
        # Embedded graph representations and curve features
        zi_sv.append(zis)
        magnitude.append(curve.curve[:,:,1].max(dim=-1)[0].detach().cpu().numpy()) #curve.curve.size() == (batch, resol,2)
        
    zis = np.concatenate(zi_li, axis=0)
    has_labels_vec = np.concatenate(labels_li, axis=0)    
    zi_sv = np.concatenate(zi_sv, axis=0)
    magnitude = np.concatenate(magnitude, axis=0)

    tsne_plot(zis, has_labels_vec, zi_sv, magnitude, cur_iter)

def tsne_plot(zis, has_labels_vec, zi_sv, magnitude, cur_iter):
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    zis_viz = tsne.fit_transform(zis) # (n_samples, n_components)
    
    # tsne plot of unlabelled and labelled graphs
    df = pd.DataFrame()
    df["y"] = has_labels_vec
    df["comp-1"] = zis_viz[:, 0]
    df["comp-2"] = zis_viz[:, 1]
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 2),
                    data=df).set(title="Pretrained graph embeddings T-SNE projection")
    plt.savefig(os.path.join(log_dir, f'tsne_{cur_iter}.png'))
    plt.clf()
    
    # tsne plot of labelled data with curve features
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    zis_viz = tsne.fit_transform(zi_sv) # (n_samples, n_components)
       
    plt.figure()
    plt.scatter(zis_viz[:, 0],zis_viz[:, 1], c= magnitude, cmap = 'viridis')
    
    norm = plt.Normalize(0, 1)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'))
    cbar.set_label('Stress magnitude')
    plt.xlabel('Comp-1')
    plt.ylabel('Comp-2')
    plt.savefig(os.path.join(log_dir, f'tsne_stress_magn_{cur_iter}.png'))
    plt.clf()
    plt.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    