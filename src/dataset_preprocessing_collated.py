import os
import random
import json
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tqdm import tqdm

import sys
# sys.path.append(r'C:\Users\Marco Maurizi\Desktop\UC_Berkeley\Research\MLgeometric_lattices\Inverse_model\GraphMetaMat')

from src.generative_graph.tesellate import tesselate, untesselate, rm_double_edges, rm_redundant_nodes
from src.utils import upsample, compute_lengths, rho2r, rotations, reflections, plot_3d
from src.dataset_feats_node import get_node_feats
from src.dataset_feats_edge import get_edge_li, get_edge_index, get_edge_feats


######################### Parameters ####################################
DATASET_CFG = {
    'base_cell': 'unitcell', #'octet', #'unitcell', # unitcell, octet, tetrahedron
    'rm_redundant':True
}
RESOLUTION_CURVE = 256

skip_non_monotonic = False
# dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/marco_07052023'
# dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/marco_07232023_ETH'
# dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/marco_09052023'

# dn_dataset = '/workspace/datasets/cubic_N3_4_D5_FLEXIBLE_strain30pct'
# dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/marco_10052023_flipped'
# dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/marco_07232023_ETH'
# dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/marco_08182023_ETH'
# dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/marco_12142023_unseen'
# dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/marco_02142024_PSU'
# dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/marco_10052023'
# dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/marco_03182024_OUR'
# dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/marco_03302024_UCB'
# dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/marco_03182024_ablation'
RL_dataset = False # False
ETH_FULL_C_VECTOR = False
PSU_dataset = False # True
ETH_dataset = False
OUR_dataset = False
# dn_dataset = r'C:\Users\Marco Maurizi\Desktop\UC_Berkeley\Research\MLgeometric_lattices\Inverse_model\GraphMetaMat_inverse_design\datasets\cubic_N3_4_D5_edges_FLEXIBLE_monotonic'
# dn_dataset = r'C:\Users\Marco Maurizi\Desktop\UC_Berkeley\Research\MLgeometric_lattices\Inverse_model\GraphMetaMat_inverse_design\datasets\cubic_N3_4_D5_FLEXIBLE_strain50pct'
# dn_dataset = r'C:\Users\Marco Maurizi\Desktop\UC_Berkeley\Research\MLgeometric_lattices\Inverse_model\GraphMetaMat\datasets\cubic_N3_4_D5_FLEXIBLE_strain30pct'
# dn_out = os.path.join(dn_dataset, f'preprocessed{"_all" if not skip_non_monotonic else ""}')
mix_dataset = False # True
is_monotonic = True
smoothing_curves = True #True
use_synthetic = False

noise_augm = 0
data_augm = False
#########################################################################

####### Paths ######
flipped = False
# dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/marco_11012023_RL'
dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/marco_03182024_ablation'
dn_out = os.path.join(dn_dataset, f'__preprocessed_{DATASET_CFG["base_cell"]}_{DATASET_CFG["rm_redundant"]}{"_mixed" if mix_dataset else ""}{"_augm" if data_augm else ""}{"_full" if ETH_FULL_C_VECTOR else ""}{f"_naugm{noise_augm}" if noise_augm > 0 else ""}{f"_flipped" if flipped else ""}')


################# Split dataset #######################
random.seed(123)
split_cfg = {
    # 'train':0.99,
    # 'dev':0.005,
    # 'test':0.005
    'train':0.9,
    'dev':0.05,
    # 'test':1.0,
    # 'train':0.9,
    # 'dev':0.1,
    'test':0.05
}
########################################################


def smoothing_out(data):
    data = savgol_filter(data, 21, 3)
    return data

def obj2curve_OUR(obj, cid, g):
    Z_avg = g.number_of_edges()/g.number_of_nodes()
    angle_li = []
    for nid in g.nodes():
        nid_li = list(g.neighbors(nid))
        assert nid not in nid_li
        for i, nid1 in enumerate(nid_li):
            for nid2 in nid_li[i+1:]:
                r1 = g.nodes[nid1]['coord'] - g.nodes[nid]['coord']
                r2 = g.nodes[nid2]['coord'] - g.nodes[nid]['coord']
                angle = np.arccos(np.dot(r1, r2)/(np.linalg.norm(r1)*np.linalg.norm(r2)))
                angle_li.append(angle)
    angle_li = np.array(angle_li)
    c_curve = np.stack((
        np.zeros(5),
        np.array([1.0, min(angle_li), angle_li.mean(), max(angle_li), Z_avg])
    ), axis=0)
    c = \
        {
            # 'curve': np.stack([np.arange(128), obj['Ex']*np.arange(128)], axis=-1),
            'curve': c_curve,
            'cid': cid,
            'is_monotonic': True
        }
    return c

def obj2curve_ETH(obj, cid):
    # print(obj['Es'])
    if ETH_FULL_C_VECTOR:
        c = \
            {
                # 'curve': np.stack([np.arange(128), obj['Ex']*np.arange(128)], axis=-1),
                'curve': np.stack([np.arange(RESOLUTION_CURVE)] + [
                    obj['C'][i] * np.arange(RESOLUTION_CURVE)/(RESOLUTION_CURVE-1) #* obj['Es']
                    for i in range(len(obj['C']))
                ], axis=-1),
                'cid': cid,
                'is_monotonic': True
            }
    else:
        c = \
             {
                 # 'curve': np.stack([np.arange(128), obj['Ex']*np.arange(128)], axis=-1),
                'curve': np.stack([
                    np.arange(RESOLUTION_CURVE),
                    obj['C'][0] * obj['Es'] * np.arange(RESOLUTION_CURVE)
                ], axis=-1),
                'cid': cid,
                'is_monotonic': True
            }
    return c

def obj2curve(obj, cid):
    stress = np.array(obj['transmissibility' if PSU_dataset else 'stress'])
    # if PSU_dataset:
    #     # stress = np.fft.ifft(np.power(10.0, stress[::10]/20.0)).real
    #     stress[stress < -40.0] = -40.0
    if smoothing_curves:
        stress = smoothing_out(stress)
    if 'max_strain' in obj and 'strain' not in obj:
        strain = np.arange(0, len(stress)) / (len(stress) - 1) * obj['max_strain']
    elif 'strain' in obj:
        strain = np.array(obj['strain']) # 0.3 * np.arange(len(obj['strain']))/(len(obj['strain'])-1) # np.array(obj['strain'])
    elif PSU_dataset:
        assert len(stress.shape) == 1
        strain = np.arange(0, len(stress)) / (len(stress) - 1)
    else:
        print('Strain not found !')
        assert False
    if PSU_dataset:
        x = strain
        y = stress
    else:
        strain[0] = 0.0
        stress[0] = 0.0
        x = strain.clip(min=0)
        y = stress.clip(min=0)
        strain[-1] = 0.3
    curve = np.stack((x, y), axis=-1)
    curve = upsample(curve, RESOLUTION_CURVE)
    c = {
        'curve': curve,
        'cid': cid,
        'is_monotonic': is_monotonic
    }
    return c

def obj2nxgraph_dummy(obj, gid):
    assert 'nodal_positions' not in obj
    g = nx.Graph()
    g.add_nodes_from(list(range(8)))
    g.nodes[0]['coord'] = np.array([-1,-1,-1])
    g.nodes[1]['coord'] = np.array([-1,-1,1])
    g.nodes[2]['coord'] = np.array([-1,1,-1])
    g.nodes[3]['coord'] = np.array([1,-1,-1])
    g.nodes[4]['coord'] = np.array([-1,1,1])
    g.nodes[5]['coord'] = np.array([1,-1,1])
    g.nodes[6]['coord'] = np.array([1,1,-1])
    g.nodes[7]['coord'] = np.array([1,1,1])
    g.add_edges_from([
        (0,1),(0,2),(4,1),(4,2),
        (3,5),(3,6),(7,5),(7,6),
        (0,3),(1,5),(2,6),(4,7),
    ])
    g.graph = {'gid': gid, 'cell_type': 'unknown', 'Es': 1.0, 'name': 'None', 'rho': 1.0}
    for eid in g.edges():
        g.edges[eid]['radius'] = 1.0
        g.edges[eid]['Es'] = 1.0
    return g

def obj2nxgraph(obj, gid, add_noise=False):
    g = nx.Graph()
    g.add_nodes_from(list(range(len(obj['nodal_positions']))))
    g.add_edges_from([[u - 1, v - 1] for (u, v) in obj['connectivity']])
    for nid, coord in enumerate(obj['nodal_positions']):
        coord = 2 * (np.array([float(val) for val in coord]).reshape(-1) - 0.5)
        if add_noise:
            coord = coord + np.clip(0.1 * np.random.randn(*coord.shape), a_min=-0.1, a_max=0.1)
        g.nodes[nid]['coord'] = coord

    g, _ = rm_double_edges(g, eps=1e-12)
    g, _ = rm_redundant_nodes(g)

    nid_li = list(g.nodes())
    for nid_u, nid_v in g.edges():
        coord_u = g.nodes[nid_u]['coord']
        coord_v = g.nodes[nid_v]['coord']
        coord_min = np.minimum(coord_u, coord_v)
        coord_max = np.maximum(coord_u, coord_v)
        nid_to_merge = []
        nid_to_merge_dist = []
        for nid in nid_li:
            coord = g.nodes[nid]['coord']
            if nid not in {nid_u, nid_v} and \
                all(coord > coord_min) and \
                all(coord < coord_max):
                ray = coord - coord_u
                ray_v = coord_v - coord_u
                norm_ray = np.linalg.norm(ray)
                norm_ray_v = np.linalg.norm(ray_v)
                assert norm_ray < norm_ray_v
                cos = np.dot(ray_v,ray)/(norm_ray*norm_ray_v)
                assert cos > 0
                if np.abs(1-cos) < 1e-10:
                    nid_to_merge.append(nid)
                    nid_to_merge_dist.append(norm_ray)
        assert len(nid_to_merge) == len(nid_to_merge_dist)
        if len(nid_to_merge) > 0:
            print('EDGE SURGERY DETECTED!')
            g.remove_edge(nid_u, nid_v)
            nid_prev = nid_u
            for i in sorted(list(range(len(nid_to_merge))), key=lambda x: nid_to_merge_dist[x]):
                nid_cur = nid_to_merge[i]
                g.add_edge(nid_prev, nid_cur)
                nid_prev = nid_cur
            g.add_edge(nid_prev, nid_v)

    space_group = obj.get('space_group', 'unknown')
    Es = obj.get('Es', 0.0)

    g = compute_lengths(g)
    if 'radius' in obj.keys():
        rho = None
        r = obj['radius']
    elif 'rho' in obj.keys():
        rho = obj['rho']
        if add_noise:
            rho = rho + np.clip(0.02 * np.random.randn(1), a_min=-0.02, a_max=0.02)
        r = rho2r(rho, g, base_cell=DATASET_CFG['base_cell'])
    else:
        print('Set relative density to 0.05')
        rho = 0.05
        r = rho2r(rho, g, base_cell=DATASET_CFG['base_cell'])

    g.graph = {'gid': gid, 'cell_type': space_group, 'Es': Es, 'name': 'None', 'rho': rho}
    for eid in g.edges():
        g.edges[eid]['radius'] = r
        g.edges[eid]['Es'] = Es

    return g


class DatasetCollatedConverter:
    def __init__(self, dn_dataset, dataset_cfg):
        # Pre-training
        if ETH_dataset:
            dn_ft = os.path.join(dn_dataset, 'Unifying_design_ETH')
        else:
            # Fine-tuning
            dn_ft = os.path.join(dn_dataset, 'Fine_tuning_data')
        print('load dataset...')
        g_dict_ft, c_dict_ft, obj_dict_ft, mapping_pairs_ft = self._get_dataset_ft_wrapper(dn_ft, dataset_cfg)
        if mix_dataset:
            random.shuffle(mapping_pairs_ft)
        print('split dataset...')
        self.split2dataset = \
            self._split_dataset_ft(g_dict_ft, c_dict_ft, obj_dict_ft, mapping_pairs_ft, data_augm, dataset_cfg)

    def _get_dataset_ft_wrapper(self, dn_ft, dataset_cfg):
        if use_synthetic:
            assert False
            # g_dict_ft, c_dict_ft, mapping_pairs_ft = self._get_dataset_synthetic()
        else:
            g_dict_ft, c_dict_ft, obj_dict_ft, mapping_pairs_ft = {}, {}, {}, []
            offset = {'offset_g': 0, 'offset_c': 0}
            fp_li = [fp for fp in os.listdir(dn_ft)]
            fp_li = sorted(fp_li, reverse=True)
            if flipped:
                fp_li = fp_li[::-1]
            for fp in fp_li:
                pn_ft = os.path.join(dn_ft, fp)
                g_dict_subsample, c_dict_subsample, obj_dict_subsample, mapping_pairs_subsample = \
                    self._get_dataset_ft(pn_ft, **offset, **dataset_cfg)
                g_dict_ft.update(g_dict_subsample)
                c_dict_ft.update(c_dict_subsample)
                obj_dict_ft.update(obj_dict_subsample)
                offset = {'offset_g': len(g_dict_ft), 'offset_c': len(c_dict_ft)}
                mapping_pairs_ft.extend(mapping_pairs_subsample)

        for c in list(c_dict_ft.values())[:500]:
            x, y = c['curve'][:, 0], c['curve'][:, 1]  # -y_mean)
            plt.plot(x, y, linewidth=0.75, alpha=0.8)  # , color=f(cy_max))
        plt.savefig(dn_dataset + '/tmp01.png')

        return g_dict_ft, c_dict_ft, obj_dict_ft, mapping_pairs_ft

    def obj2graph(self, obj, gid, base_cell, rm_redundant, add_noise=False, **kwargs):
        g = obj2nxgraph_dummy(obj, gid) if RL_dataset else obj2nxgraph(obj, gid, add_noise)
        if ETH_dataset:
            for nid in g.nodes():
                g.nodes[nid]['coord'] = (g.nodes[nid]['coord'] + 1) / 2
            # coord_li = np.stack([g.nodes[nid]['coord'] for nid in g.nodes()], axis=0)
            # print(coord_li.min(axis=0), ',', coord_li.max(axis=0))
            if base_cell in ['unitcell', 'octet']:
                g = tesselate(g, start='octet', end=base_cell, rm_redundant=rm_redundant)
            elif base_cell in ['tetrahedron']:
                g = tesselate(g, start='octet', end=base_cell, rm_redundant=rm_redundant)
            else:
                assert False
        else:
            g = untesselate(g, start='unitcell', end=base_cell, rm_redundant=rm_redundant)

        if len(list(nx.selfloop_edges(g))) != 0:
            print('WARNING: g contains self-loops!')
        g.remove_edges_from(nx.selfloop_edges(g)) # remove self loops
        edge_li = get_edge_li(g)
        edge_index = get_edge_index(edge_li)
        node_feats = get_node_feats(g, edge_index)
        edge_feats = get_edge_feats(g, edge_li)
        edge_index = edge_index.detach().cpu().numpy()

        g.graph['node_feats'] = node_feats
        g.graph['edge_index'] = edge_index
        g.graph['edge_feats'] = edge_feats

        assert edge_index.shape[1] == edge_feats.shape[0] == 2 * g.number_of_edges()
        assert node_feats.shape[0] == g.number_of_nodes()
        return g

    def _get_dataset_ft(self, pn_ft, offset_g=0, offset_c=0, cutoff=0.08, base_cell='octet', rm_redundant=True):
        with open(pn_ft) as fp:
            dataset = json.load(fp)
        g_dict = {}
        c_dict = {}
        obj_dict = {}
        mapping_pairs = []
        for id, obj in enumerate(tqdm(dataset)):
            gid, cid = id + offset_g, id + offset_c
            g = self.obj2graph(obj, gid, base_cell, rm_redundant, add_noise=False)

            if ETH_dataset:
                curve = obj2curve_ETH(obj, cid)
            elif OUR_dataset:
                curve = obj2curve_OUR(obj, cid, g=g)
            else:
                curve = obj2curve(obj, cid)

            if skip_non_monotonic and np.sum(curve['curve'][1:] < curve['curve'][:-1]) != 0:
                continue

            g_dict[gid] = g
            c_dict[cid] = curve
            obj_dict[gid] = obj
            mapping_pairs.append((gid, cid))
        return g_dict, c_dict, obj_dict, mapping_pairs

    def _get_dataset_synthetic(self):
        from networkx.generators.classic import cycle_graph
        g_dict = {}
        c_dict = {}
        mapping_pairs = []
        num_pairs = 1000
        for i in range(num_pairs):
            k = i + 2
            g = cycle_graph(k)
            for nid in g.nodes():
                g.nodes[nid]['coord'] = np.random.rand(3).reshape(-1)
            for eid in g.edges():
                nid_start, nid_end = eid
                g.edges[eid]['length'] = np.linalg.norm(g.nodes[nid_start]['coord'] - g.nodes[nid_end]['coord'])
            for eid in g.edges():
                g.edges[eid]['radius'] = k
            g.graph = {'gid': i, 'cell_type': None, 'Es': k, 'name': 'None'}
            for eid in g.edges():
                g.edges[eid]['Es'] = k
            g_dict[i] = g
            c_dict[i] = {
                'curve': np.stack([np.arange(128), (2 ** (10 * k / num_pairs)) * np.arange(128)], axis=-1),
                'cid': i,
                'is_monotonic': is_monotonic
            }
            mapping_pairs.append((i, i))
        return g_dict, c_dict, mapping_pairs

    def _split_dataset_ft(self, g_dict, c_dict, obj_dict, mapping_pairs, data_augm, dataset_cfg):
        cur_idx = 0
        split_idx_li = []
        for split, amount in tqdm(split_cfg.items()):
            next_idx = len(mapping_pairs) if split == list(split_cfg.keys())[-1] else int(
                cur_idx + len(mapping_pairs) * amount)
            assert cur_idx != next_idx
            split_idx_li.append((split, cur_idx, next_idx))
            cur_idx = next_idx

        split2dataset = {}
        for split, cur_idx, next_idx in tqdm(split_idx_li):
            mapping_pairs_split = mapping_pairs[cur_idx:next_idx]
            gid_li_split, cid_li_split = list(zip(*mapping_pairs_split))
            g_dict_split = {gid: g_dict[gid] for gid in gid_li_split}
            c_dict_split = {cid: c_dict[cid] for cid in cid_li_split}

            # def perturb(g):
            #     g_new = deepcopy(g)
            #     g_new.node_feats = g_new.node_feats + torch.randn_like(g_new.node_feats)/10
            #     return g_new
            #
            if noise_augm > 0 and split == 'train':
                gid_cur = max(g_dict_split.keys())+1
                cid_cur = max(c_dict_split.keys())+1
                for gid, cid in mapping_pairs_split:
                    for _ in range(noise_augm):
                        g_dict_split[gid_cur] = \
                            self.obj2graph(
                                obj_dict[gid],
                                gid_cur,
                                **dataset_cfg,
                                add_noise=False
                            )
                        c_dict_split[cid_cur] = c_dict_split[cid]
                        gid_cur += 1
                        cid_cur += 1

            if data_augm and split == 'train':
                ################ Data augmentation ###############################
                if split == 'train':
                    g_li_symm = []
                    c_li_symm = []
                    for k, gid in enumerate(gid_li_split):
                        assert gid == cid_li_split[k]
                        c = c_dict_split[gid]
                        g = g_dict_split[gid]
                        # for i in range(len(g_li_augm)):
                        #     for nid in g_li_augm[i].nodes():
                        #         g_li_augm[i].nodes[nid]['coord'] = 2.0*g_li_augm[i].nodes[nid]['coord'] - 1.0
                        # assert all([g.number_of_nodes() == ga.number_of_nodes() for ga in g_li_augm])

                        g_li_augm = rotations(g)  # list of graphs
                        g_li_refl = reflections(g) # list of graphs
                        g_li_augm.extend(g_li_refl)

                        # for k, g_ in enumerate(g_li_augm):
                        #     g_tetrahedron = untesselate(g_, start='unitcell', end='tetrahedron', rm_redundant=True)
                        #     if g_tetrahedron.number_of_edges() != g_tetrahedron.number_of_nodes() - 1:
                        #         from src.utils import plot_3d_debug
                        #         plot_3d_debug(g_tetrahedron, nm='gt')
                        #         print(g_tetrahedron.number_of_edges(), g_tetrahedron.number_of_nodes())
                        #         plot_3d_debug(g, nm='g')
                        #         for i, g_ in enumerate(g_li_augm):
                        #             plot_3d_debug(g_, nm=f'g_{i}')
                        #         print(k)
                        #         assert False


                        c_li_augm = [c for _ in range(len(g_li_augm))]

                        g_li_symm.extend(g_li_augm)
                        c_li_symm.extend(c_li_augm)

                    max_gid = max(gid_li_split)
                    max_cid = max(cid_li_split)
                    assert max_gid == max_cid
                    for i, g in enumerate(g_li_symm):
                        g_dict_split[max_gid + 1 + i] = g
                        c_dict_split[max_cid + 1 + i] = c_li_symm[i]
                        mapping_pairs_split.append((max_gid + 1 + i, max_cid + 1 + i))

                    # max_gid = max(g_dict_split.keys())
                    # mapping_pairs_split = [(i, i) for i in range(max_gid + 1)]
                ##################################################################

            print(split, len(g_dict_split))
            split2dataset[split] = \
                {
                    'g_dict': g_dict_split,
                    'c_dict': c_dict_split,
                    'mapping_pairs': mapping_pairs_split
                }
        return split2dataset


def save_dataset_ft(g_dict, c_dict, mapping_pairs, dn_out, split, base_cell='octet', **kwargs):
    pn_match = os.path.join(dn_out, split)
    pn_graphs = os.path.join(dn_out, split, "graphs")
    pn_curves = os.path.join(dn_out, split, "curves")
    os.mkdir(pn_match)
    os.mkdir(pn_graphs)
    os.mkdir(pn_curves)

    # Save curves
    for cid, c in tqdm(c_dict.items()):
        with open(os.path.join(pn_curves, f'{cid}.pkl'), 'wb') as fp:
            pickle.dump(c, fp)

    # save graphs
    for gid, g in tqdm(g_dict.items()):
        with open(os.path.join(pn_graphs, f'{gid}.gpkl'), 'wb') as f:
            pickle.dump(g, f)

        if not use_synthetic and not ETH_dataset:
            g_tetrahedron = untesselate(g, start=base_cell, end='tetrahedron', rm_redundant=True)
            # if g_tetrahedron.number_of_nodes() > 4:
            #     pickle.dump({'g':g}, open(
            #         '/home/derek/Documents/MaterialSynthesis/problem_unc.pkl', 'wb'))
            #     g_octet = untesselate(g, start=base_cell, end='octet', rm_redundant=True)
            #     pickle.dump({'g':g_octet}, open(
            #         '/home/derek/Documents/MaterialSynthesis/problem_oct.pkl', 'wb'))
            #     pickle.dump({'g':g_tetrahedron}, open(
            #         '/home/derek/Documents/MaterialSynthesis/problem_tet.pkl', 'wb'))
            assert g_tetrahedron.number_of_nodes() <= 4, \
                f'tetrahedron has {g_tetrahedron.number_of_nodes()} nodes; graph has {g.number_of_nodes()} nodes'
            # assert g_tetrahedron.number_of_edges() == g_tetrahedron.number_of_nodes() - 1
            with open(os.path.join(pn_graphs, f'{gid}_polyhedron.gpkl'), 'wb') as f:
                pickle.dump(g_tetrahedron, f)

    # save mapping
    with open(os.path.join(pn_match, 'mapping.tsv'), 'w+') as fp:
        fp.writelines(['\t'.join([str(id) for id in mapping]) + '\n' for mapping in mapping_pairs])


def main(dataset_cfg):
    assert dn_out != dn_dataset
    # os.system(f'mkdir {dn_out}')
    if not os.path.exists(dn_out):
        os.mkdir(dn_out)

    print('Data processing start...')
    dataset_converter = DatasetCollatedConverter(dn_dataset, dataset_cfg)
    split2dataset = dataset_converter.split2dataset

    print('Data saving start...')
    print('Fine-tuning data saving start...')
    for split, data_dict in split2dataset.items():
        save_dataset_ft(**data_dict, dn_out=dn_out, split=split, **dataset_cfg)

    print('Pre-training data saving start...')
    print('Data saving done.')


def main_ETH_fast(dataset_cfg):
    assert dn_out != dn_dataset
    # os.system(f'mkdir {dn_out}')
    os.mkdir(dn_out)

    print('Data processing start...')
    dn_ft = os.path.join(dn_dataset, 'Unifying_design_ETH')
    fp, = os.listdir(dn_ft)
    pn_ft = os.path.join(dn_ft, fp)

    with open(pn_ft) as fp:
        dataset = json.load(fp)
    n_pairs = len(dataset)
    indices_li = list(range(n_pairs))

    if mix_dataset:
        random.shuffle(indices_li)

    cur_idx = 0
    split_idx_li = []
    for split, amount in tqdm(split_cfg.items()):
        next_idx = n_pairs if split == list(split_cfg.keys())[-1] else int(
            cur_idx + n_pairs * amount)
        split_idx_li.append((split, cur_idx, next_idx))
        cur_idx = next_idx

    idx2split = {}
    for split, cur_idx, next_idx in tqdm(split_idx_li):
        for idx in indices_li[cur_idx:next_idx]:
            idx2split[idx] = split

    for split in split_cfg.keys():
        pn_match = os.path.join(dn_out, split)
        pn_graphs = os.path.join(dn_out, split, "graphs")
        pn_curves = os.path.join(dn_out, split, "curves")
        os.mkdir(pn_match)
        os.mkdir(pn_graphs)
        os.mkdir(pn_curves)

    base_cell = 'unitcell'
    rm_redundant = True
    for id, obj in enumerate(tqdm(dataset)):
        gid, cid = id, id
        g = obj2nxgraph(obj, gid)
        if ETH_dataset:
            assert not OUR_dataset
            for nid in g.nodes():
                g.nodes[nid]['coord'] = (g.nodes[nid]['coord'] + 1) / 2
            # coord_li = np.stack([g.nodes[nid]['coord'] for nid in g.nodes()], axis=0)
            # print(coord_li.min(axis=0), ',', coord_li.max(axis=0))
            if base_cell in ['unitcell', 'octet']:
                g = tesselate(g, start='octet', end=base_cell, rm_redundant=rm_redundant)
            elif base_cell in ['tetrahedron']:
                g = tesselate(g, start='octet', end=base_cell, rm_redundant=rm_redundant)
            else:
                assert False
            curve = obj2curve_ETH(obj, cid)
        else:
            assert not OUR_dataset
            g = untesselate(g, start='unitcell', end=base_cell, rm_redundant=rm_redundant)
            curve = obj2curve(obj, cid)

        edge_li = get_edge_li(g)
        edge_index = get_edge_index(edge_li)
        node_feats = get_node_feats(g, edge_index)
        edge_feats = get_edge_feats(g, edge_li)
        edge_index = edge_index.detach().cpu().numpy()

        g.graph['node_feats'] = node_feats
        g.graph['edge_index'] = edge_index
        g.graph['edge_feats'] = edge_feats

        assert edge_index.shape[1] == edge_feats.shape[0] == 2 * g.number_of_edges()
        assert node_feats.shape[0] == g.number_of_nodes()

        if skip_non_monotonic and np.sum(curve['curve'][1:] < curve['curve'][:-1]) != 0:
            continue

        split = idx2split[id]
        pn_match = os.path.join(dn_out, split)
        pn_graphs = os.path.join(dn_out, split, "graphs")
        pn_curves = os.path.join(dn_out, split, "curves")

        # Save curves
        with open(os.path.join(pn_curves, f'{cid}.pkl'), 'wb') as fp:
            pickle.dump(curve, fp)

        # save graphs
        with open(os.path.join(pn_graphs, f'{gid}.gpkl'), 'wb') as f:
            pickle.dump(g, f)

        if not use_synthetic and not ETH_dataset:
            g_tetrahedron = untesselate(g, start=base_cell, end='tetrahedron', rm_redundant=True)
            assert g_tetrahedron.number_of_nodes() <= 4
            assert g_tetrahedron.number_of_edges() == g_tetrahedron.number_of_nodes() - 1
            with open(os.path.join(pn_graphs, f'{gid}_polyhedron.gpkl'), 'wb') as f:
                pickle.dump(g_tetrahedron, f)

        # save mapping
        with open(os.path.join(pn_match, 'mapping.tsv'), 'a+') as fp:
            fp.writelines([f'{id}\t{id}\n'])


if __name__ == '__main__':
    main(DATASET_CFG)
