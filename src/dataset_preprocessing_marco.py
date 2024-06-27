import os
import csv
import pickle

import networkx as nx
import numpy as np
import random
random.seed(123)

from scipy import stats
from utils import is_mutually_exclusive, set_new_max, upsample
from tqdm import tqdm
from bidict import bidict
from itertools import groupby
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

do_plot = False#True
do_clean = True
is_monotonic = True
dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/marco_02212023'
# dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/marco_03062023_diffbasematerials'
# dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/marco_03062023_diffbasematerials_mixedwithETHgraphs'
# fn = 'cubic_N3_D5_set1_graphs_curves.json'
fn = 'cubic_N3_D5_graphs_curves.json'
# fn = 'cubic_N3_4_D5_mixed_graphs_curves.json'
# fn = 'cubic_N3_4_D5_mixed_with_crystalsETH_graphs_curves.json'

mix_dataset = True
suffix = '_mixed' if mix_dataset else ''
suffix += '_clean' if do_clean else ''
dn_out = f'{dn_dataset}/preprocessed{suffix}'
# dn_out = 'temp'
split_cfg = {
    'train':0.8,
    'dev':0.1,
    'test':0.1
} # TODO
assert sum(split_cfg.values()) == 1

def compute_stats(g_dict, c_dict, mapping_pairs):
    stats_g = compute_stats_g(g_dict.values())
    stats_c = compute_stats_c(c_dict.values())
    return stats_g, stats_c

def compute_stats_g(g_li):
    all_coords = []
    all_lengths = []
    all_radiuses = []
    for g in g_li:
        all_coords.extend([x for nid in g.nodes() for x in g.nodes[nid]['coord']])
        all_lengths.extend([g.edges[eid]['length'] for eid in g.edges()])
        all_radiuses.extend([g.edges[eid]['radius'] for eid in g.edges()])
    stats_g = {
        'unit_cell_size': max(all_coords),
        'length_stats': {
            'mean': np.mean(np.array(all_lengths)),
            'std': np.std(np.array(all_lengths))+1e-12
        },
        'radius_stats': {
            'mean': np.mean(np.array(all_radiuses)),
            'std': np.std(np.array(all_radiuses))+1e-12
        }
    }
    return stats_g

def compute_stats_c(c_li):
    '''
    Returns:
        unit_cell_size = size of unit cell
        length_stats = {'mean': u, 'std': std} across whole dataset
        radius_stats = {'mean': u, 'std': std} across whole dataset
    '''
    stats_c = {'mean': [], 'log2mean': [], 'log10mean': []}
    for c in c_li:
        c_max = np.max(c['curve'], axis=0) + 1e-8
        stats_c = \
            {
                'mean': stats_c['mean'] + [c_max],
                'log2mean': stats_c['log2mean'] + [np.log2(c_max)],
                'log10mean': stats_c['log10mean'] + [np.log10(c_max)]
            }
    stats_c = {k: np.mean(np.stack(v, axis=0), axis=0) for k, v in stats_c.items()}
    return stats_c

def skip_pair(g, c):
    return np.sum(c['curve'][1:] < c['curve'][:-1]) != 0

def collect_curve_stats(c_li):
    c_norm_li, c_scale_x, c_scale_y = [], [], []
    for v in c_li:
        c = v['curve']
        c_norm = c / np.max(c, axis=0)
        cx_max, cy_max = np.max(c, axis=0)
        c_scale_x.append(cx_max)
        c_scale_y.append(cy_max)
        c_norm_li.append(c_norm)
    return c_norm_li, c_scale_x, c_scale_y

def plot_curves(c_norm_li, c_scale_x, c_scale_y):
    norm = mpl.colors.Normalize(vmin=min(c_scale_y), vmax=max(c_scale_y))
    cmap = cm.gnuplot
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    f = lambda x: m.to_rgba(x)
    i = 0
    plt.clf()
    for c_norm, cx_max, cy_max in zip(c_norm_li, c_scale_x, c_scale_y):
        # if np.sum(c_norm[1:] < c_norm[:-1]) == 0:
        plt.plot(c_norm[:,0], c_norm[:,1])#, color=f(cy_max))
        i += 1
        # if i > 10:
        #     break
        print(c_norm.max(axis=0))
    plt.savefig('tmp00.png')
    plt.clf()

    # heatmap, xedges, yedges = \
    #     np.histogram2d(c_scale_x, c_scale_y, bins=10)
    # # Construct arrays for the anchor positions of the 16 bars.
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # plt.imshow(heatmap.T, extent=extent, origin='lower')
    # plt.savefig('tmp01.png')
    assert False

class DatasetConverterV1():
    def __init__(self, dn):
        self.dn = dn
        self.cid_mapping = {}

    def get_dataset_wrapper(self):
        g_dict, c_dict, mapping_pairs = \
            self._get_dataset()
        stats_g, stats_c = compute_stats(g_dict, c_dict, mapping_pairs)
        for gid, g in g_dict.items():
            g.graph.update(stats_g)
        for cid, c in c_dict.items():
            c['stats'] = stats_c
        return g_dict, c_dict, mapping_pairs

    def _get_dataset(self):
        g_dict, c_dict, mapping_pairs = {}, {}, []

        # loop through all gene csv files
        obj_li = json.load(open(os.path.join(self.dn, fn)))

        curve_valid_max = \
            np.array([
                max(obj2curve(obj, -1)['curve'][:,0]) for obj in obj_li])

        for id, obj in enumerate(obj_li):
            g = obj2nxgraph(obj, id)
            curve = obj2curve(obj, id, min(curve_valid_max))
            if do_clean and skip_pair(g, curve):
                continue
            print(curve['curve'][:, 0].max())
            g_dict[id] = g
            c_dict[id] = curve
            mapping_pairs.append((id, id))

        if do_plot:
            plot_curves(*collect_curve_stats(c_dict.values()))

        return g_dict, c_dict, mapping_pairs

def save_dataset(g_dict, c_dict, mapping_pairs, pn_match, pn_graphs, pn_curves):
    # save graphs
    for gid, g in g_dict.items():
        nx.write_gpickle(g, os.path.join(pn_graphs,f'{gid}.gpkl'))

    # save curves
    for cid, c in c_dict.items():
        with open(os.path.join(pn_curves,f'{cid}.pkl'), 'wb') as fp:
            pickle.dump(c, fp)

    # save mapping
    with open(os.path.join(pn_match,'mapping.tsv'), 'w+') as fp:
        fp.writelines(['\t'.join([str(id) for id in mapping])+'\n' for mapping in mapping_pairs])

def get_curve_unlabelled_dataset(stats_c, c_norm_li, c_scale_x, c_scale_y, pn_curves_unlabelled, num_samples=1000):
    c_scale_x_minmax = min(c_scale_x), max(c_scale_x)
    c_scale_y_minmax = min(c_scale_y), max(c_scale_y)

    cid = 0
    for c_norm in tqdm(c_norm_li):
        c_scale_sampled = np.stack((
            np.random.uniform(*c_scale_x_minmax, size=num_samples),
            np.random.uniform(*c_scale_y_minmax, size=num_samples)), axis=1)
        c_norm_augmented = \
            np.expand_dims(c_norm, 0) * np.expand_dims(c_scale_sampled, 1)
        for curve in c_norm_augmented:
            c = {
                'curve': curve,
                'cid': cid,
                'is_monotonic': is_monotonic,
                'stats' : stats_c
            }
            with open(os.path.join(pn_curves_unlabelled,f'{cid}.pkl'), 'wb') as fp:
                pickle.dump(c, fp)
            cid += 1



def main():
    assert dn_out != dn_dataset
    os.system(f'mkdir {dn_out}')

    print('data processing start...')
    dataset_converter = DatasetConverterV1(dn_dataset)
    g_dict, c_dict, mapping_pairs = dataset_converter.get_dataset_wrapper()
    print('data saving start...')

    if mix_dataset:
        random.shuffle(mapping_pairs)

    cur_idx = 0
    split_idx_li = []
    for i, (split, amount) in enumerate(split_cfg.items()):
        next_idx = len(mapping_pairs) if i == len(split_cfg)-1 else int(cur_idx + amount * len(mapping_pairs))
        split_idx_li.append((split, cur_idx, next_idx))
        cur_idx = next_idx

    pn_graphs_unlabelled = os.path.join(dn_out, "unlabelled_graphs")
    pn_curves_unlabelled = os.path.join(dn_out, "unlabelled_curves")
    pn_curves_unlabelled_more = os.path.join(dn_out, "unlabelled_curves_more")
    pn_curves_unlabelled_even_more = os.path.join(dn_out, "unlabelled_curves_more")
    os.system(f'mkdir {pn_graphs_unlabelled}')
    os.system(f'mkdir {pn_curves_unlabelled_more}')
    os.system(f'mkdir {pn_curves_unlabelled_even_more}')

    for split, cur_idx, next_idx in split_idx_li:
        pn_match = os.path.join(dn_out, split)
        pn_graphs = os.path.join(dn_out, split, "graphs")
        pn_curves = os.path.join(dn_out, split, "curves")
        os.system(f'mkdir {pn_match}')
        os.system(f'mkdir {pn_graphs}')
        os.system(f'mkdir {pn_curves}')

        mapping_pairs_split = mapping_pairs[cur_idx:next_idx]
        gid_li_split, cid_li_split = list(zip(*mapping_pairs_split))
        g_dict_split = {gid:g for gid,g in g_dict.items() if gid in gid_li_split}
        c_dict_split = {cid:c for cid,c in c_dict.items() if cid in cid_li_split}
        save_dataset(g_dict_split, c_dict_split, mapping_pairs_split, pn_match, pn_graphs, pn_curves)

        if split == 'train':
            stats_c = list(c_dict_split.values())[0]['stats']
            c_norm_li, c_scale_x, c_scale_y = \
                collect_curve_stats(c_dict_split.values())
            # get_curve_unlabelled_dataset(
            #     stats_c, c_norm_li, c_scale_x, c_scale_y,
            #     pn_curves_unlabelled, num_samples=1000)
            #
            # for _ in range(10*len(c_norm_li)):
            #     c_norm1, c_norm2 = random.sample(c_norm_li, 2)
            #     c_norm_new = (c_norm1 + upsample(c_norm2, c_norm1.shape[0])) / 2
            #     c_norm_li.append(c_norm_new)
            # c_norm_li = [c_norm for c_norm in c_norm_li]
            # get_curve_unlabelled_dataset(
            #     stats_c, c_norm_li, c_scale_x, c_scale_y,
            #     pn_curves_unlabelled_more, num_samples=100)

            for _ in range(1000*len(c_norm_li)):
                c_norm1, c_norm2 = random.sample(c_norm_li, 2)
                c_norm_new = (c_norm1 + upsample(c_norm2, c_norm1.shape[0])) / 2
                c_norm_li.append(c_norm_new)
            c_norm_li = [c_norm for c_norm in c_norm_li]
            get_curve_unlabelled_dataset(
                stats_c, c_norm_li, c_scale_x, c_scale_y,
                pn_curves_unlabelled_more, num_samples=1)

    print('success!')
#
# def augment():
#     import glob
#     dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/SyntheticData_05102022_simple/preprocessed/train'
#     dn_out = f'{dn_dataset}_augmented'
#     assert dn_out != dn_dataset
#     # if os.path.isdir(dn_out):
#     #     os.system(f'cp -r {dn_dataset} {dn_out}')
#
#     print('data processing start...')
#     g_dict = \
#         {
#             int(os.path.split(pn)[1].split('.gpkl')[0]): nx.read_gpickle(pn)
#             for pn in glob.glob(os.path.join(dn_dataset, 'graphs', '*'))
#         }
#     c_dict = \
#         {
#             int(os.path.split(pn)[1].split('.pkl')[0]): pickle.load(open(pn, 'rb')) # TODO: close the file after
#             for pn in glob.glob(os.path.join(dn_dataset, 'curves', '*'))
#         }
#     mapping_pairs = \
#         {
#             int(k):int(v) for k,v in
#             csv.reader(open(os.path.join(dn_dataset, 'mapping.tsv')), delimiter="\t")
#         }
#     mapping_pairs_clustered = split_mapping_pairs(g_dict, mapping_pairs)
#     augment_helper(g_dict, c_dict, mapping_pairs_clustered)
#     print('data saving start...')

def split_mapping_pairs(g_dict, mapping_pairs):
    from collections import defaultdict
    mapping_pairs_clustered = defaultdict(lambda: {'mapping_pair_li': [], 'radius_li': []})
    for gid, cid in mapping_pairs.items():
        name = g_dict[gid].graph['name']
        gene_id, radius_id = name.strip('.csv').split('_')
        mapping_pairs_clustered[gene_id]['mapping_pair_li'].append((gid, cid))
        mapping_pairs_clustered[gene_id]['radius_li'].append(radius_id)
    return mapping_pairs_clustered

def augment_helper(g_dict, c_dict, mapping_pairs_clustered):
    for gid, mapping_pairs_dict in mapping_pairs_clustered.items():
        radius_li = [int(r.strip('R')) for r in mapping_pairs_dict['radius_li']]
        graph_li, curve_li = [], []
        for gid, cid in mapping_pairs_dict['mapping_pair_li']:
            graph_li.append(g_dict[gid])
            curve_li.append(c_dict[cid]['curve'])
        c_li = interpolate_curves(radius_li, curve_li)
        # g_li = interpolate_graphs(radius_id_li, graph_li)

def interpolate_curves(radius_li, curve_li, k=50, resolution=256):
    from utils import upsample
    curve_li = [upsample(c, resolution) for c in curve_li]


    import scipy as sp
    import scipy.interpolate
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF

    X_train = np.arange(len(radius_li)) / len(radius_li)
    y_train = np.array([max(c[:,0]) for c in curve_li])
    X_test = np.arange(k) / k
    # kernel = \
    #     1 * RBF(
    #         length_scale=10.0,
    #         length_scale_bounds=(1e-1, 1e3))
    # gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    # gaussian_process.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))
    f = scipy.interpolate.interp1d(X_train.reshape(-1,1), y_train.reshape(-1,1))
    y_max_interp = f(X_test.reshape(-1,1)).reshape(-1) # gaussian_process.predict(X_test.reshape(-1,1)).reshape(-1)

    x = np.concatenate([c[:,0] for c in curve_li])
    y = np.concatenate([np.array([i/len(curve_li)]*len(c)) for i, c in enumerate(curve_li)])*max(x)
    z = np.concatenate([c[:,1] for c in curve_li])

    # spline = sp.interpolate.Rbf(x, y, z, function='Gaussian', smooth=0.1)#, function='thin_plate', smooth=5, episilon=5)
    spline = sp.interpolate.interp2d(x,y,z,kind='linear')

    curve_li_interpolated = []
    for i, y_max in enumerate(y_max_interp):
        x_new = np.arange(resolution)/resolution*y_max
        y_new = np.array([i/k])*max(x) # np.array([i/k]*resolution)*max(x)
        z_new = spline(x_new, y_new)
        curve_li_interpolated.append(np.stack([x_new, z_new], axis=-1))


    # X_train = np.array(radius_li) / max(radius_li)
    # X_test = np.arange(k) / k
    # curves_augmented = np.zeros(shape=(resolution, 2, k))
    # for i in tqdm(range(resolution)):
    #     for j in range(2):
    #         y_train = np.array([c[i,j] for c in curve_li])
    #         kernel = \
    #             1 * RBF(
    #                 length_scale=max(1.0,float(np.mean(y_train))),
    #                 length_scale_bounds=(1e-2, max(1e2, 5*max(y_train))))
    #         gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    #         gaussian_process.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))
    #         curves_augmented[i,j] = gaussian_process.predict(X_test.reshape(-1,1)).reshape(-1)


    import matplotlib
    matplotlib.use('GTK3Agg')
    for c in curve_li:
        plt.plot(c[:,0], c[:,1])
    plt.plot(y_max_interp, upsample(np.array([max(c[:,1]) for c in curve_li]), k))
    # plt.yscale('log')
    plt.show()

    for c in curve_li_interpolated:
        plt.plot(c[:,0], c[:,1])
    plt.plot(y_max_interp, upsample(np.array([max(c[:,1]) for c in curve_li]), k))
    # plt.yscale('log')
    plt.show()
    assert False


def obj2curve(obj, cid, curve_valid_max=None):
    stress = np.array(obj['stress'])
    strain = np.arange(0, len(stress))/(len(stress)-1)*obj['max_strain']
    x = strain.clip(min=0)
    y = stress.clip(min=0)#/max(stress)*np.log(max(stress))
    curve = np.stack((x,y), axis=-1)
    if curve_valid_max is not None:
        curve = set_new_max(curve, curve_valid_max)
    c = {
        'curve': curve,
        'cid': cid,
        'is_monotonic': is_monotonic
    }
    return c

def obj2nxgraph(obj, gid):
    g = nx.Graph()
    g.add_nodes_from(list(range(len(obj['nodal_positions']))))
    g.add_edges_from([[u-1, v-1] for (u,v) in obj['connectivity']])
    for nid, coord in enumerate(obj['nodal_positions']):
        g.nodes[nid]['coord'] = np.array([float(val) for val in coord]).reshape(-1)
    for eid in g.edges():
        nid_start, nid_end = eid
        g.edges[eid]['radius'] = 1.0
        g.edges[eid]['length'] = np.linalg.norm(g.nodes[nid_start]['coord'] - g.nodes[nid_end]['coord'])

    space_group = obj.get('space_group', 'unknown')
    g.graph = {'gid':gid, 'cell_type':space_group, 'name':'None'}
    return g

if __name__ == '__main__':
    main()
    # augment()
