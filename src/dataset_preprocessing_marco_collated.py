import os
import random
import re
import json
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from utils import set_new_max, upsample
from tqdm import tqdm

skip_non_monotonic = False
dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/marco_05192023'
dn_out = os.path.join(dn_dataset, f'preprocessed{"_all" if not skip_non_monotonic else ""}')
mix_dataset = True
is_monotonic = True
random.seed(123)

split_cfg = {
    'train':0.8,
    'dev':0.1,
    'test':0.1
}

def compute_stats(g_dict, c_dict):
    stats_g = compute_stats_g(g_dict.values())
    stats_c = compute_stats_c(c_dict.values())
    return stats_g, stats_c

def obj2curve(obj, cid):
    stress = np.array(obj['stress'])
    if 'strain' in obj:
        strain = np.array(obj['strain'])
    else:
        strain = np.arange(0, len(stress))/(len(stress)-1) * obj['max_strain']
    # print(min(strain), max(strain), min(stress), max(stress))
    x = np.array(strain.clip(min=0).tolist() + [0.3])
    y = np.array(stress.clip(min=0).tolist() + [stress[-1]])
    curve = np.stack((x,y), axis=-1)
    c = {
        'curve': curve,
        'cid': cid,
        'is_monotonic': is_monotonic
    }
    return c

def obj2nxgraph(obj, gid, rho=None):
    g = nx.Graph()
    g.add_nodes_from(list(range(len(obj['nodal_positions']))))
    g.add_edges_from([[u-1, v-1] for (u,v) in obj['connectivity']])
    for nid, coord in enumerate(obj['nodal_positions']):
        g.nodes[nid]['coord'] = np.array([float(val) for val in coord]).reshape(-1)
    for eid in g.edges():
        nid_start, nid_end = eid
        g.edges[eid]['length'] = np.linalg.norm(g.nodes[nid_start]['coord'] - g.nodes[nid_end]['coord'])

    if 'radius' in obj.keys():
        assert rho is None
        r = obj['radius']
    else:
        assert rho is not None
        L = 10.0
        Lsum = sum([g.edges[eid]['length'] for eid in g.edges()])
        r = np.sqrt(rho * L ** 3 / (np.pi * Lsum))

    for eid in g.edges():
        g.edges[eid]['radius'] = r

    space_group = obj.get('space_group', 'unknown')
    Es = obj.get('Es', 0.0)
    g.graph = {'gid':gid, 'cell_type':space_group, 'Es':Es, 'name':'None'}

    for nid in g.nodes():
        g.nodes[nid]['Es'] = Es
    return g

def compute_stats_g(g_li):
    all_coords = []
    all_lengths = []
    all_radiuses = []
    all_Es = []
    for g in g_li:
        all_coords.extend([x for nid in g.nodes() for x in g.nodes[nid]['coord']])
        all_lengths.extend([g.edges[eid]['length'] for eid in g.edges()])
        all_radiuses.extend([g.edges[eid]['radius'] for eid in g.edges()])
        all_Es.append([g.graph['Es']])
    stats_g = {
        'unit_cell_size': max(all_coords),
        'length_stats': {
            'mean': np.mean(np.array(all_lengths)),
            'std': np.std(np.array(all_lengths))+1e-12
        },
        'radius_stats': {
            'mean': np.mean(np.array(all_radiuses)),
            'std': np.std(np.array(all_radiuses))+1e-12
        },
        'Es_stats': {
            'mean': np.mean(np.array(all_Es)),
            'std': np.std(np.array(all_Es))+1e-12
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

class DatasetCollatedConverter:
    def __init__(self, dn_dataset):
        dn_ft = os.path.join(dn_dataset, 'Fine_tuning_data')
        g_dict_ft, c_dict_ft, mapping_pairs_ft = self._get_dataset_ft_wrapper(dn_ft)
        if mix_dataset:
            random.shuffle(mapping_pairs_ft)
        self.split2dataset = \
            self._split_dataset_ft(g_dict_ft, c_dict_ft, mapping_pairs_ft)

        self.curves_unlabelled = \
            self._get_curves_unlabelled(self.split2dataset['train'])
        # self.graphs_unlabelled = \
        #     self._get_graphs_unlabelled_wrapper(os.path.join(dn_dataset, 'Pre_training_data'))
        self.graphs_unlabelled = None

    def _get_curves_unlabelled(self, dataset_ft_train, num_samples=10):
        curves_unlabelled = list(dataset_ft_train['c_dict'].values())
        c_stats = curves_unlabelled[0]['stats']
        c_scale_y_minmax = \
            min([max(c['curve'][:,1]) for c in curves_unlabelled]), \
            max([max(c['curve'][:,1]) for c in curves_unlabelled])

        cid = 0
        c_norm_augmented = []

        for c in curves_unlabelled:
            c_norm = c['curve']
            c_norm[:, 1] /= max(c_norm[:, 1])
            c_norm_augmented.append(c_norm)

        for _ in range(3):
            for _ in range(10*len(curves_unlabelled)):
                c_norm1, c_norm2 = random.sample(c_norm_augmented, 2)
                c_norm_new = (c_norm1 + upsample(c_norm2, c_norm1.shape[0])) / 2
                c_norm_augmented.append(c_norm_new)

        curves_unlabelled_augmented = []
        for c_norm in tqdm(c_norm_augmented):
            c_scale_sampled = np.stack((
                np.ones(num_samples),
                np.random.uniform(*c_scale_y_minmax, size=num_samples)), axis=1)
            c_norm_augmented = \
                np.expand_dims(c_norm, 0) * np.expand_dims(c_scale_sampled, 1)
            for curve in c_norm_augmented:
                c = {
                    'curve': curve,
                    'cid': cid,
                    'is_monotonic': is_monotonic,
                    'stats': c_stats
                }
                curves_unlabelled_augmented.append(c)
                cid += 1
        return curves_unlabelled_augmented

    def _get_graphs_unlabelled_wrapper(self, dn_graphs):
        graphs_unlabelled = []
        offset = 0
        for fp in os.listdir(dn_graphs):
            pn_graphs = os.path.join(dn_graphs, fp)
            graphs_unlabelled_subsample = self._get_graphs_unlabelled(pn_graphs, offset)
            graphs_unlabelled.extend(graphs_unlabelled_subsample)
            offset = len(graphs_unlabelled)
        return offset

    def _get_graphs_unlabelled(self, pn_graphs, offset=0):
        pass

    def _get_dataset_ft_wrapper(self, dn_ft):
        g_dict_ft, c_dict_ft, mapping_pairs_ft = {}, {}, []
        offset = {'offset_g':0, 'offset_c':0}
        for fp in os.listdir(dn_ft):
            pn_ft = os.path.join(dn_ft, fp)
            g_dict_subsample, c_dict_subsample, mapping_pairs_subsample = \
                self._get_dataset_ft(pn_ft, **offset)
            g_dict_ft.update(g_dict_subsample)
            c_dict_ft.update(c_dict_subsample)
            offset = {'offset_g':len(g_dict_subsample), 'offset_c':len(c_dict_subsample)}
            mapping_pairs_ft.extend(mapping_pairs_subsample)

        # # curve postprocessing
        new_max = 0.3 # min([max(c['curve'][:,0]) for c in c_dict_ft.values()])
        print('new_max:', new_max)
        for k, v in list(c_dict_ft.items()):
            v['curve'] = set_new_max(v['curve'], new_max)
            c_dict_ft[k] = v

        for c in c_dict_ft.values():
            # if np.sum(c_norm[1:] < c_norm[:-1]) == 0:
            x, y = c['curve'][:,0], c['curve'][:,1]
            plt.plot(x,y,alpha=0.75,linewidth=0.75)#, color=f(cy_max))
        plt.savefig('/home/derek/Documents/MaterialSynthesis/data/marco_05192023/tmp01.png')
        plt.clf()

        # collect statistics
        stats_g, stats_c = \
            compute_stats(
                g_dict_ft, #split2dataset['train']['g_dict'],
                c_dict_ft, #split2dataset['train']['c_dict'],
            )
        for gid, g in g_dict_ft.items():
            g.graph.update(stats_g)
        for cid, c in c_dict_ft.items():
            c['stats'] = stats_c

        return g_dict_ft, c_dict_ft, mapping_pairs_ft

    def _get_dataset_ft(self, pn_ft, offset_g=0, offset_c=0, cutoff=0.08):
        is_rho = re.findall(r'rho\d+', pn_ft)
        if len(is_rho) == 0:
            rho = 0.05
        else:
            assert len(is_rho) == 1
            rho = float(is_rho[0][3:]) / 100

        with open(pn_ft) as fp:
            dataset = json.load(fp)
        g_dict = {}
        c_dict = {}
        mapping_pairs = []
        for id, obj in enumerate(dataset):
            gid, cid = id+offset_g, id+offset_c
            g = obj2nxgraph(obj, gid, rho=rho)
            curve = obj2curve(obj, cid)
            if curve['curve'][-2,0] < 0.275:
                continue
            if skip_non_monotonic and np.sum(curve['curve'][1:] < curve['curve'][:-1]) != 0:
                continue
            g_dict[gid] = g
            c_dict[cid] = curve
            mapping_pairs.append((gid, cid))
        return g_dict, c_dict, mapping_pairs

    def _split_dataset_ft(self, g_dict, c_dict, mapping_pairs):
        cur_idx = 0
        split_idx_li = []
        for i, (split, amount) in enumerate(split_cfg.items()):
            next_idx = len(mapping_pairs) if i == len(split_cfg)-1 else int(cur_idx + amount * len(mapping_pairs))
            split_idx_li.append((split, cur_idx, next_idx))
            cur_idx = next_idx

        split2dataset = {}
        for split, cur_idx, next_idx in split_idx_li:
            mapping_pairs_split = mapping_pairs[cur_idx:next_idx]
            gid_li_split, cid_li_split = list(zip(*mapping_pairs_split))
            g_dict_split = {gid: g for gid, g in g_dict.items() if gid in gid_li_split}
            c_dict_split = {cid: c for cid, c in c_dict.items() if cid in cid_li_split}
            split2dataset[split] = \
                {
                    'g_dict': g_dict_split,
                    'c_dict': c_dict_split,
                    'mapping_pairs': mapping_pairs_split
                }


        from collections import defaultdict
        split2max_li = defaultdict(list)
        for split, dataset in split2dataset.items():
            for c in dataset['c_dict'].values():
                # if np.sum(c_norm[1:] < c_norm[:-1]) == 0:
                x, y = c['curve'][:,0], c['curve'][:,1]
                split2max_li['all'].append(y[-1])
                split2max_li[split].append(y[-1])
        for split, max_li in split2max_li.items():
            plt.hist(np.array(max_li), bins=20, alpha=0.5, label=split)
        plt.legend()
        plt.savefig('/home/derek/Documents/MaterialSynthesis/data/marco_05192023/tmp02.png')
        plt.clf()
        for split, max_li in split2max_li.items():
            plt.hist(np.log10(np.array(max_li)), bins=20, alpha=0.5, label=split)
        plt.legend()
        plt.savefig('/home/derek/Documents/MaterialSynthesis/data/marco_05192023/tmp03.png')
        return split2dataset

def main():
    assert dn_out != dn_dataset
    os.system(f'mkdir {dn_out}')

    print('data processing start...')
    dataset_converter = DatasetCollatedConverter(dn_dataset)
    split2dataset = dataset_converter.split2dataset
    curves_unlabelled = dataset_converter.curves_unlabelled

    print('data saving start...')
    for split, data_dict in split2dataset.items():
        save_dataset_ft(**data_dict, dn_out=dn_out, split=split)
    save_dataset_pt(curves_unlabelled, dn_out)
    print('data saving done.')

def save_dataset_ft(g_dict, c_dict, mapping_pairs, dn_out, split):
    pn_match = os.path.join(dn_out, split)
    pn_graphs = os.path.join(dn_out, split, "graphs")
    pn_curves = os.path.join(dn_out, split, "curves")
    os.system(f'mkdir {pn_match}')
    os.system(f'mkdir {pn_graphs}')
    os.system(f'mkdir {pn_curves}')

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


def save_dataset_pt(c_li, dn_out):
    pn_graphs_unlabelled = os.path.join(dn_out, "unlabelled_graphs")
    os.system(f'mkdir {pn_graphs_unlabelled}')
    # pn_curves_unlabelled = os.path.join(dn_out, "unlabelled_curves")
    # os.system(f'mkdir {pn_curves_unlabelled}')

    # save curves
    # for cid, c in enumerate(c_li):
    #     with open(os.path.join(pn_curves_unlabelled,f'{cid}.pkl'), 'wb') as fp:
    #         pickle.dump(c, fp)

if __name__ == '__main__':
    main()
    # augment()
