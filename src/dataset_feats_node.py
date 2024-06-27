import torch
import numpy as np
import networkx as nx
from torch_scatter import scatter_max, scatter_min, scatter_mean, scatter_std
from torch_geometric.utils import degree

from src.generative_graph.tesellate import reflect_points
NODE_FEAT_CFG = [
    'x','y','z','x_proj','y_proj','z_proj','radius','deg','min_deg','max_deg','mean_deg','std_deg'
]

def local_degree_profile(num_nodes, edge_index):
    row, col = edge_index

    deg = degree(row, num_nodes, dtype=torch.float)
    deg_col = deg[col]

    min_deg, _ = scatter_min(deg_col, row, dim_size=num_nodes)
    min_deg[min_deg > 10000] = 0
    max_deg, _ = scatter_max(deg_col, row, dim_size=num_nodes)
    max_deg[max_deg < -10000] = 0
    mean_deg = scatter_mean(deg_col, row, dim_size=num_nodes)
    std_deg = scatter_std(deg_col, row, dim_size=num_nodes)

    deg, min_deg, max_deg, mean_deg, std_deg = \
        deg.view(-1).cpu().numpy(), min_deg.view(-1).cpu().numpy(), \
        max_deg.view(-1).cpu().numpy(), mean_deg.view(-1).cpu().numpy(), \
        std_deg.view(-1).cpu().numpy()
    return deg, min_deg, max_deg, mean_deg, std_deg

def get_node_feats_manual(g, coord_force=np.array([0,0,0.5]), lattice_width=1, eps=1e-9, radius_default=0.0):
    x_li, y_li, z_li, x_proj_li, y_proj_li, z_proj_li, radius_li = [], [], [], [], [], [], []
    for nid in g.nodes():
        x, y, z = g.nodes[nid]['coord']
        x_proj, y_proj, z_proj = project_into_triangular_prism(g.nodes[nid]['coord'])
        radius = []
        for nid_c in nx.neighbors(g, nid):
            radius.append(g.edges[nid, nid_c]['radius'] if 'radius' in g.edges[nid, nid_c] else radius_default)
        radius = np.array(radius).mean()
        x_li.append(x)
        y_li.append(y)
        z_li.append(z)
        x_proj_li.append(x_proj)
        y_proj_li.append(y_proj)
        z_proj_li.append(z_proj)
        radius_li.append(radius)

    x_li, y_li, z_li, x_proj_li, y_proj_li, z_proj_li, radius_li = \
        np.array(x_li), np.array(y_li), np.array(z_li), \
        np.array(x_proj_li), np.array(y_proj_li), np.array(z_proj_li), np.array(radius_li)

    return x_li, y_li, z_li, x_proj_li, y_proj_li, z_proj_li, radius_li

def get_node_feats_ldp(g, edge_index):
    node_feats = local_degree_profile(g.number_of_nodes(), edge_index)
    return node_feats

def get_node_feats(g, edge_index, manual_args=None, skip_ldp=False):
    if manual_args is None:
        manual_args = {}
    x, y, z, x_proj, y_proj, z_proj, radius = get_node_feats_manual(g, **manual_args)
    if skip_ldp:
        deg, min_deg, max_deg, mean_deg, std_deg = \
            None, None, None, None, None
    else:
        deg, min_deg, max_deg, mean_deg, std_deg = get_node_feats_ldp(g, edge_index=edge_index)
    var_dict = locals()
    return np.stack([var_dict[feat] for feat in NODE_FEAT_CFG], axis=-1)

def project_into_triangular_prism(coord):
    plane1 = np.array([0,0,1])
    plane2 = np.array([0,1,0])
    plane3 = np.array([1,0,0])
    plane4 = np.array([-1,-1,0])
    for plane_vector in [plane1, plane2, plane3, plane4]:
        if np.dot(coord, plane_vector) < 0:
            coord = reflect_points(np.expand_dims(coord, axis=0), plane_vector).squeeze(axis=0)
    return coord
