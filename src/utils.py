import os
import scipy.interpolate as interpolate
from datetime import datetime
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.validation import make_valid
from tensorboardX import SummaryWriter
from matplotlib import cm

import sys
import numpy as np
import copy
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score
from matplotlib import cm
from tensorboardX import SummaryWriter
import torch
import torch_geometric.nn.norm as norm
import matplotlib.colors as mcolors
cmap = list(mcolors.TABLEAU_COLORS)

from src.config import log_dir, args

# writer = SummaryWriter(log_dir)
PLUS_MINUS = u"\u00B1"

import shutil
if os.path.exists(os.path.join(log_dir, 'reward')):
    shutil.rmtree(os.path.join(log_dir, 'reward'))
if os.path.exists(os.path.join(log_dir, 'loss')):
    shutil.rmtree(os.path.join(log_dir, 'loss'))
tboard = SummaryWriter(log_dir)

class GraphEncoder(torch.nn.Module):
    def __init__(self, encoder, pooler):
        super(GraphEncoder, self).__init__()
        self.encoder = encoder
        self.pooler = pooler

    def forward(self, graph):
        emb_nodes, emb_edges = \
            self.encoder(
                graph.feats_node, graph.feats_edge, graph.edge_index,
                graph_node_index=graph.graph_node_index,
                graph_edge_index=graph.graph_edge_index
            )
        out = \
            self.pooler(emb_nodes, emb_edges, graph)
        return out

class CObj():
    def __init__(self, c=None, c_magnitude=None, c_shape=None, c_magnitude_std=None, c_shape_std=None, c_UBLB=None):
        self.c = c
        self.c_UBLB = c_UBLB
        self.c_magnitude = c_magnitude
        self.c_shape = c_shape
        self.c_magnitude_std = c_magnitude_std
        self.c_shape_std = c_shape_std

    def update(self, indices_to_update, c_obj):
        if c_obj is None:
            try:
                assert not (True in indices_to_update)
            except:
                asdf = 0
                assert False
        else:
            assert self.c is None
            assert self.c_UBLB is None
            if self.c_magnitude is not None:
                self.c_magnitude[indices_to_update] = c_obj.c_magnitude[indices_to_update]
            if self.c_magnitude_std is not None:
                self.c_magnitude_std[indices_to_update] = c_obj.c_magnitude_std[indices_to_update]
            if self.c_shape is not None:
                self.c_shape[indices_to_update] = c_obj.c_shape[indices_to_update]
            if self.c_shape_std is not None:
                self.c_shape_std[indices_to_update] = c_obj.c_shape_std[indices_to_update]
        return self

class DoubleDict():
    def __init__(self):
        self.l2r = {}
        self.r2l = {}

    def __len__(self):
        return len(self.l2r)

    def add(self, k, v):
        if k in self.l2r:
            assert self.l2r[k] == v
        if v in self.r2l:
            assert self.r2l[v] == k, f'{self.r2l, k, v}'
        self.l2r[k] = v
        self.r2l[v] = k

def to_item(t):
    return t.cpu().detach().item()

def to_np(t):
    return t.cpu().detach().numpy()

def to_list(t):
    return t.cpu().detach().tolist()

def plot_3d(g, lim=None, ax=None, pn=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    node_xyz = np.array([g.nodes[nid]['coord'] for nid in g.nodes()])
    edge_li = g.edges()
    edge_xyz = \
        np.array([
            (g.nodes[nid_u]['coord'], g.nodes[nid_v]['coord'])
            for nid_u, nid_v in edge_li])

    radius = []
    for eid in edge_li:
        if 'radius' in g.edges[eid]:
            radius.append(g.edges[eid]['radius'])
        else:
            radius.append(0.1)

    ax.scatter(*node_xyz.T, s=100, ec="w")
    cmap = cm.get_cmap('gnuplot')
    for vizedge, vizradius in zip(edge_xyz, radius):
        coord = (vizedge[0] + vizedge[1]) / 2
        color_id = (np.dot(np.array([1, 1, 1]), coord) + 3)*50/6
        if color_id < 0:
            print(f'WARNING color id OoB: {color_id}')
            color_id = 0
        color_id /= 50
        plt.axis()
        ax.plot(*vizedge.T, color=cmap(color_id), lw=vizradius * 10, alpha=0.8)
    if lim is not None:
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_zlim(*lim)

    # _format_axes(ax)
    ax.grid(True)
    # fig.tight_layout()
    if pn is not None:
        plt.savefig(pn)
        plt.clf()

def plot_3d_wrapper(g_dict, c_dict, pn=None):
    g_dict.pop('Train (3)')
    g_dict.pop('Train (4)')
    fig = plt.figure(figsize=(8*(1 + len(g_dict)),8))

    assert len(g_dict) < 9
    for i, (k, g) in enumerate(sorted(g_dict.items())):
        # if k=='True':
        #     continue
        ax = fig.add_subplot(100 + 10*(1 + len(g_dict)) + 1 + i, projection="3d")
        if 'C' in g.graph:
            ax.set_title(f"{k} Graph (rho={g.graph['rho']:.5f}, C={g.graph['C']:.5f}, n={g.graph['n']:.5f})")
        else:
            ax.set_title(f"{k} Graph (rho={g.graph['rho']:.5f})")
        plot_3d(g, ax=ax)

    ax = fig.add_subplot(100 + 10*(1 + len(g_dict)) + (1 + len(g_dict)))
    ax.set_title('Curve')
    plot_curve(c_dict, ax=ax)
    fig.tight_layout()
    if pn is not None:
        plt.savefig(pn)
        plt.clf()
        plt.close()


def plot_3d_debug(g, nm=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    node_xyz = np.array([g.nodes[nid]['coord'] for nid in g.nodes()])
    edge_li = g.edges()
    edge_xyz = \
        np.array([
            (g.nodes[nid_u]['coord'], g.nodes[nid_v]['coord'])
            for nid_u, nid_v in edge_li])
    ax.scatter(*node_xyz.T, s=100, ec="w")
    cmap = cm.get_cmap('gnuplot')
    for vizedge in edge_xyz:
        coord = (vizedge[0] + vizedge[1]) / 2
        color_id = (np.dot(np.array([1, 1, 1]), coord) + 3)*50/6
        if color_id < 0:
            print(f'WARNING color id OoB: {color_id}')
            color_id = 0
        color_id /= 50
        plt.axis()
        ax.plot(*vizedge.T, color=cmap(color_id), alpha=0.5)
    if nm is None:
        plt.show()
    else:
        plt.savefig(f'/home/derek/Documents/MaterialSynthesis/{nm}.png')
    plt.clf()
    plt.close(fig)

def _format_axes(ax):
    """Visualization options for the 3D axes."""
    # Turn gridlines off
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_axis_off()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

def plot_curve(c_dict, pn=None, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    cid = 0

    if 'True Curve' in c_dict:
        cmap_ = cm.get_cmap('RdYlGn')
        if 'digitize_cfg' in args['dataset'] and args['dataset']['digitize_cfg'] is not None:
            c_digital = np.digitize(c_dict['True Curve'][0], args['dataset']['digitize_cfg']['bins'])
            x = 0.3*np.arange(len(c_digital))/(len(c_digital)-1)
            blocksize = int(x.shape[0] / args['dataset']['digitize_cfg']['n_freq'])
            c_digital = np.median(c_digital.reshape(-1, blocksize), axis=-1).astype(int)
            c_digital = c_digital.repeat(blocksize)

            for threshold in x[::blocksize]:
                ax.axvline(threshold, color='black', lw=1, alpha=0.7)
            for threshold in args['dataset']['digitize_cfg']['bins']:
                ax.axhline(threshold, color='black', lw=2, alpha=0.7)

            for h in set(c_digital):
                ax.fill_between(
                    x, 0, 1, where=(c_digital==h),
                    color=cmap_(h*256/len(set(c_digital))), alpha=0.1, transform=ax.get_xaxis_transform()
                )

    for label, (c, c_LB, c_UB) in sorted(c_dict.items(), reverse=True):
        if label == 'FwdModel(True Graph)':
            continue
        strain_li = 0.3*np.arange(len(c))/(len(c)-1)

        if '[skip]' in f'{label}':
            color = 'black'
            ax.plot(strain_li, c, alpha=0.2, linestyle='-', color=color)
        else:
            color = cmap[cid]
            cid += 1
            ax.plot(strain_li, c, alpha=1.0, linestyle='-', color=color, label=label)
        if c_LB is not None:
            assert c_UB is not None
            ax.fill_between(strain_li, c_UB.reshape(-1), c_LB.reshape(-1), alpha=0.25, color=color)
            # try:
            #     ax.fill_between(strain_li, c_UB.reshape(-1), c_LB.reshape(-1), alpha=0.25)
            # except:
            #     print('asdfadsfas')

    # ax.x_label('Strain')
    # ax.y_label('Stress')
    ax.grid(False)
    ax.set_xlabel(r'$ \varepsilon $ (%)')
    ax.set_ylabel(r'$ \sigma $ (MPa)')
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')
    ax.legend()
    if pn is not None:
        plt.savefig(pn)
        plt.clf()

def plot_magnitude(magnitude_pred_all, magnitude_true_all, i=None, pn=None, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    z = np.polyfit(magnitude_true_all, magnitude_pred_all, 1)
    r_squared = r2_score(magnitude_true_all, magnitude_pred_all)

    ax.scatter(
        magnitude_true_all, magnitude_pred_all, color='gray',
        marker='x', label=("y=%.6fx+(%.6f) - $R^2$=%.6f" % (z[0], z[1], r_squared)))
    if i is not None:
        ax.scatter(
            magnitude_true_all[[i]], magnitude_pred_all[[i]], color='blue', marker='o')
    ax.set_xlabel('magnitude_true (z-score)')
    ax.set_ylabel('magnitude_pred (z-score)')
    ax.grid()
    ax.plot(sorted(magnitude_true_all), sorted(magnitude_true_all), "y--")
    ax.legend()
    if pn is not None:
        plt.savefig(pn)
        plt.clf()

def write_plot_loss(dn, loss, max_update, plot_suffix = 'train'):
    if type(loss) == list:
        loss = np.array(loss)
    print('Loss saving...')
    np.savetxt(os.path.join(dn, f'loss_{plot_suffix}.txt'), loss)    
    print('Loss saved!')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    iterations = np.linspace(0,max_update, num = len(loss))
    ax.plot(iterations, loss, 'k', label = 'Training loss')        
    # Set axes labels
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')        
    # ax.legend()    
    fig.tight_layout()
    plt.savefig(os.path.join(log_dir, f'loss_{plot_suffix}.png'))
    plt.clf()
    plt.close()

def write_to_log(dn, msg):
    print(msg)
    with open(os.path.join(dn, 'log.txt'), 'a') as fp:
        fp.write(msg + '\n')


def get_pbar(total_steps, init_step=0):
    pbar = tqdm(total=total_steps, dynamic_ncols=True, desc='overall', file=sys.stderr)
    pbar.n = init_step

def standardize_norm(vec, stats):
    return (vec - stats['mean']) / stats['std']

# def upsample(seq, resolution):
#     t_old = np.arange(seq.shape[0]) / (seq.shape[0]-1)
#     t_new = np.arange(resolution) / (resolution-1)
#     return interpolate.interp1d(t_old, seq, kind='linear', axis=0)(t_new)

def upsample(seq, resolution, kind='cubic'):
    seq_new = np.zeros((resolution,2))
    t_old = seq[:,0]
    t_new = np.arange(resolution) / (resolution-1) * seq[:,0].max()
    seq_new[:,1:] = np.stack([
        interpolate.interp1d(t_old, seq[:,i+1], kind=kind, fill_value='extrapolate', axis=0)(t_new)
        for i in range(seq.shape[1]-1)], axis=1)
    seq_new[:,0] = t_new
    return seq_new

def set_new_max(seq, new_max):
    resolution = seq.shape[0]
    t_old = np.arange(resolution) / (resolution-1)
    t_new = np.arange(resolution) / (resolution-1) * new_max/seq[:,0].max()
    return interpolate.interp1d(t_old, seq, kind='linear', axis=0)(t_new)

def is_mutually_exclusive(s1, s2):
    return len(s1.intersection(s2)) == 0

def get_mape(x1, x2, eps=1e-12):
    mape = np.mean(np.abs(x2-x1)/(np.abs(x2)+eps))
    return mape

def get_nmae(x1, x2):        
    nmae = np.mean(np.abs(x2-x1)/(np.abs(np.max(x2) - np.min(x2))))
    return nmae

def get_mae(x1, x2):
    mae = np.mean(np.abs(x1-x2))
    return mae

def get_mse(x1, x2):
    mse = np.mean((x1-x2)**2)
    return mse

def get_rmse(x1, x2):
    rmse = np.sqrt(np.mean((x1-x2)**2))
    return rmse

def get_jaccard_index_monotonic(y1, y2, eps=1e-12):
    union_area = sum([max(y1_elt, y2_elt) for y1_elt, y2_elt in zip(y1, y2)])
    inter_area = sum([min(y1_elt, y2_elt) for y1_elt, y2_elt in zip(y1, y2)])
    jaccard_index = inter_area/(union_area+eps)
    return jaccard_index

def get_shape(curve, eps=1e-12):
    curve_shape = curve.clone()
    curve_shape[:,0] /= max(curve_shape[:,0]) + eps
    return curve_shape

def get_jaccard_index(x1, y1, x2, y2):
    p = make_valid(Polygon(list(zip(x1, y1))))
    q = make_valid(Polygon(list(zip(x2, y2))))
    jaccard_index = p.intersection(q).area / p.union(q).area
    return jaccard_index

class MLP(nn.Module):
    def __init__(self, dims, dropout=None, use_norm=False, use_skip=True, act='ELU', gnn_batch=False):
        super(MLP, self).__init__()
        self.gnn_batch = gnn_batch
        self.linear_li = nn.ModuleList([
            nn.Linear(dim1, dim2)
            for (dim1, dim2) in zip(dims, dims[1:])
        ])
        self.norm_li = nn.ModuleList([
            (norm.BatchNorm(dim2) if self.gnn_batch else nn.BatchNorm1d(dim2))
            if use_norm else None for (dim1, dim2) in zip(dims, dims[1:])
        ])
        self.use_skip = use_skip
        self.use_norm = use_norm
        self.dropout = None if dropout is None else nn.Dropout(p=dropout)
        self.act = getattr(nn, act)()

    def forward(self, X, gnn_batch=None):
        for i, (linear, norm) in enumerate(zip(self.linear_li, self.norm_li)):
            X_out = linear(X)
            if self.use_skip and X_out.shape[-1] == X.shape[-1]:
                X = X_out + X
            else:
                X = X_out
            if i != len(self.linear_li) - 1:
                X = self.act(X)
                if self.use_norm:
                    X_shape = X.shape
                    if self.gnn_batch:
                        assert gnn_batch is not None
                        X = norm(X.view(-1, X.shape[-1]), batch=gnn_batch).view(*X_shape)
                    else:
                        X = norm(X.view(-1, X.shape[-1])).view(*X_shape)
                if self.dropout is not None:
                    X = self.dropout(X)
        return X

def mean(li):
    return sum(li)/len(li)

def get_stats(arr):
    return np.array([np.min(arr), np.max(arr), np.mean(arr), np.std(arr)])

def get_angle_wrapper(coord1, coord2, coord3):
    u = coord1-coord2
    v = coord3-coord2
    return get_angle(u, v)

def get_angle(u, v):
    angle = np.arccos(u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v)+1e-12))
    return angle

def plot(XY_dict, std_dict=None, pn=None, title=''):
    plt.clf()
    # plt.title(f'Gene: {gene_type}{" - leaked" if g_hash in g_hash_li else " - test"}')
    for Y_label, (X, Y, line_style) in XY_dict.items():
        plt.plot(X, Y, alpha=0.5, linestyle=line_style, label=Y_label)
        if std_dict is not None and Y_label in std_dict:
            plt.fill_between(X, Y+std_dict[Y_label], Y-std_dict[Y_label], alpha=0.25, label=Y_label)
        # print(f'{Y_label}:\t{Y}')
    plt.legend()
    plt.title(title)
    if pn is None:
        plt.show()
    else:
        plt.savefig(pn)
        plt.close()

def to_numpy(tnsr):
    return tnsr.cpu().detach().numpy()

def within_eps(a, b, eps):
    return np.abs(a-b) <= eps

def get_face_vec(coord, unit_cell_size):
    '''
    Returns:
        3-D array:
            -1 -> intersecting with 0 axis
            +0 -> not intersecting
            +1 -> intersecting with unit_cell_size axis
        Notice:
            |AUB| = |A| + |B| - |A∩B|
            |A| = sum(abs(A))
            |A∩B| = sum(A == B)
    '''
    face_vec = []
    eps = unit_cell_size/100
    for x in coord:
        if within_eps(x, -unit_cell_size, eps):
            face_vec.append(-1)
        elif within_eps(x, unit_cell_size, eps):
            face_vec.append(1)
        else:
            face_vec.append(0)
    return np.array(face_vec)

def compute_lengths(g):
    for eid in g.edges():
        nid_start, nid_end = eid
        g.edges[eid]['length'] = np.linalg.norm(g.nodes[nid_start]['coord'] - g.nodes[nid_end]['coord'])
    return g

# def rho2r(rho, g, base_cell=None, L=10.0):
#     if base_cell == 'unitcell':
#         L = L/2
#     elif base_cell == 'octet':
#         L = L
#     elif base_cell == 'tetrahedron':
#         assert False, 'tetrahedron does not work'
#     else:
#         assert False, f'unrecognized base cell: {base_cell}'
#     Lsum = sum([L * g.edges[eid]['length'] for eid in g.edges()])
#     r = np.sqrt(rho * (L ** 3) / (np.pi * Lsum))
#     return r

def rho2r(rho, g, base_cell=None, L=10.0):               
    L = 2.0 # if coords in [-1,1]
    Lsum = sum([g.edges[eid]['length'] for eid in g.edges()])
    r = np.sqrt(rho * (L ** 3) / (np.pi * Lsum))
    return r

def rho3r(rho_old, rho_new, r):
    r_new = torch.sqrt(torch.tensor(r, device=rho_old.device)**2/rho_old*rho_new).cpu().tolist()
    return r_new

def get_optimizer(optimizer_name, optimizer_args, model):
    optimizer = getattr(torch.optim, optimizer_name)(params=model.parameters(), **optimizer_args)
    return optimizer

################### Rotations and reflections ###############################

def rotation_matrix(u, theta=90):
    '''Rotation by an angle theta around the axis u = (ux, uy, uz)'''
    # Normalize u to unit vector
    u = np.array(u)
    norm_u = np.linalg.norm(u)
    u = u / norm_u
    ux, uy, uz = u
    theta = np.radians(theta)
    c = np.cos(theta)
    s = np.sin(theta)
    R = [[c + ux ** 2 * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
         [uy * ux * (1 - c) + uz * s, c + uy ** 2 * (1 - c), uy * uz * (1 - c) - ux * s],
         [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz ** 2 * (1 - c)]]
    return np.array(R)

def rotations(g, L=1.0):
    g_li = []
    axes = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    for u in axes:
        for angle in [90]:#, 180, 270]:
            graph = copy.deepcopy(g)
            R = rotation_matrix(u, theta=angle)
            R[np.abs(R) < 1e-10] = 0.0
            nid_li = list(graph.nodes())
            for nid in nid_li:
                x = np.dot(R, np.array(graph.nodes[nid]['coord']).reshape(3, -1))[:, 0]
                x[np.abs(x) < 1e-10] = 0.0
                x[x < -1 + 1e-10] = -1.0
                x[x > 1 - 1e-10] = 1.0
                graph.nodes[nid]['coord'] = x

            # x = np.array([graph.nodes[nid]['coord'] for nid in nid_li]).reshape(3,-1)
            # assert x.shape[0] == 3
            # x = np.dot(R,x)
            # x[np.abs(x) < 1e-10] = 0.0
            # assert x.shape[0] == 3
            # for count, nid in enumerate(nid_li):
            #     # graph.nodes[nid]['coord'] = (x[:,count] + 0.5)*L
            #     graph.nodes[nid]['coord'] = x[:,count]
            g_li.append(graph)
    return g_li


def reflection_matrix(n):
    '''n is the normal vector of the mirror plane: (3,1)
        T is the mirror transf., also called the Householder Transformation'''
    n = np.array(n)
    norm_n = np.linalg.norm(n)
    n = n / norm_n
    n = n.reshape((3, 1))
    return np.eye(3, 3) - 2 * np.dot(n, n.T)


def reflections(g, L=1.0):
    g_li = []
    axes = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    for n in axes:
        graph = copy.deepcopy(g)
        T = reflection_matrix(n)
        T[np.abs(T) < 1e-10] = 0.0
        nid_li = list(graph.nodes())
        for nid in nid_li:
            x = np.dot(T, np.array(graph.nodes[nid]['coord']).reshape(3, -1))[:, 0]
            x[np.abs(x) < 1e-10] = 0.0
            x[x < -1 + 1e-10] = -1.0
            x[x > 1 - 1e-10] = 1.0
            graph.nodes[nid]['coord'] = x

        # x = np.array([graph.nodes[nid]['coord'] for nid in nid_li]).reshape(3,-1)
        # assert x.shape[0] == 3
        # x = np.dot(T,x)
        # assert x.shape[0] == 3
        # for count, nid in enumerate(nid_li):
        #     graph.nodes[nid]['coord'] = x[:,count]
        g_li.append(graph)
    return g_li
