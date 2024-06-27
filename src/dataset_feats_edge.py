import torch
import numpy as np
EDGE_FEAT_CFG = [
    'radius','l2_dist','l2_dist_xy','l2_dist_z','num_adjacent_cells'
]

def get_edge_li(g):
    edge_li = list([nid_u, nid_v] for (nid_u, nid_v) in g.edges())
    return edge_li

def get_edge_index(edge_li):
    if len(edge_li) == 0:
        edge_index = torch.empty(2, 0, dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_li, dtype=torch.long).transpose(0, 1)
        edge_index = \
            torch.stack([
                torch.cat([edge_index[0], edge_index[1]], dim=-1),
                torch.cat([edge_index[1], edge_index[0]], dim=-1)], dim=0)
    return edge_index

def get_edge_feats(g, edge_li, lattice_width=1, eps=1e-9, radius_default=0.0):
    edge_feats = []

    radius_li, l2_dist_li, l2_dist_xy_li, l2_dist_z_li, num_adjacent_cells_li = \
        [], [], [], [], []

    for nid_u, nid_v in edge_li:
        edge_vec = g.nodes[nid_u]['coord'] - g.nodes[nid_v]['coord']
        l2_dist = np.linalg.norm(edge_vec)
        l2_dist_xy = np.linalg.norm(edge_vec[:2])
        l2_dist_z = np.abs(edge_vec[2])
        radius = g.edges[nid_u, nid_v]['radius'] if 'radius' in g.edges[nid_u, nid_v] else radius_default
        num_adjacent_cells_u = np.sum(np.abs(g.nodes[nid_u]['coord']) > lattice_width/2 - eps)
        num_adjacent_cells_v = np.sum(np.abs(g.nodes[nid_v]['coord']) > lattice_width/2 - eps)
        num_adjacent_cells = min(num_adjacent_cells_u, num_adjacent_cells_v)
        # if radius is None:
        #     edge_feats.append(torch.tensor([l2_dist, l2_dist_xy, l2_dist_z, num_adjacent_cells], dtype=torch.float))
        # else:
        radius_li.append(radius)
        l2_dist_li.append(l2_dist)
        l2_dist_xy_li.append(l2_dist_xy)
        l2_dist_z_li.append(l2_dist_z)
        num_adjacent_cells_li.append(num_adjacent_cells)

    radius, l2_dist, l2_dist_xy, l2_dist_z, num_adjacent_cells = \
        np.array(radius_li), np.array(l2_dist_li), np.array(l2_dist_xy_li), \
        np.array(l2_dist_z_li), np.array(num_adjacent_cells_li)
    var_dict = locals()
    edge_feats = np.stack([var_dict[feat] for feat in EDGE_FEAT_CFG], axis=-1)
    edge_feats = np.concatenate([edge_feats, edge_feats], axis=0)
    return edge_feats

