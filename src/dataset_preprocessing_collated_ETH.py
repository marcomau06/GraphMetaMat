import os
import random
import re
import json
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from utils import plot_3d
from generative_graph.tesellate import tesselate, untesselate, rm_redundant_nodes

from copy import deepcopy
from utils import set_new_max, upsample
from tqdm import tqdm
from numpy.fft import fft

skip_non_monotonic = False
# dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/marco_07052023'
# dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/marco_07232023_ETH'
#dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/marco_07252023_ETH'
#dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/marco_07252023_ETH'
dn_dataset = '/home/derek/Documents/MaterialSynthesis/data/marco_08182023_ETH'
# dn_dataset = r'C:\Users\Marco Maurizi\Desktop\UC_Berkeley\Research\MLgeometric_lattices\Inverse_model\GraphMetaMat_inverse_design\datasets\cubic_N3_4_D5_edges_FLEXIBLE_monotonic'
# dn_dataset = r'C:\Users\Marco Maurizi\Desktop\UC_Berkeley\Research\MLgeometric_lattices\Inverse_model\GraphMetaMat_inverse_design\datasets\cubic_N3_4_D5_FLEXIBLE_strain50pct'
# dn_dataset = r'C:\Users\Marco Maurizi\Desktop\UC_Berkeley\Research\MLgeometric_lattices\Inverse_model\GraphMetaMat_inverse_design\datasets\cubic_N3_4_D5_FLEXIBLE_strain30pct'
# dn_out = os.path.join(dn_dataset, f'preprocessed{"_all" if not skip_non_monotonic else ""}')
dn_out = os.path.join(dn_dataset, 'preprocessed_mixed_Es')
mix_dataset = True
is_monotonic = True
smoothing_curves = True

use_synthetic = False

random.seed(123)
split_cfg = {
    'train':0.8,
    'dev':0.1,
    'test':0.1
}

# Unit cell size
L = 10.0

def smoothing_out(data):
    data = savgol_filter(data, 21, 3)
    return data


def obj2nxgraph(obj, gid):
    g = nx.Graph()
    g.add_nodes_from(list(range(len(obj['nodal_positions']))))
    g.add_edges_from([[u-1, v-1] for (u,v) in obj['connectivity']])
    for nid, coord in enumerate(obj['nodal_positions']):
        g.nodes[nid]['coord'] = np.array([float(val) for val in coord]).reshape(-1)
    for eid in g.edges():
        nid_start, nid_end = eid
        g.edges[eid]['length'] = np.linalg.norm(g.nodes[nid_start]['coord'] - g.nodes[nid_end]['coord'])

    if 'radius' in obj.keys():
        rho = None
        r = obj['radius']
    elif 'rho' in obj.keys():       
        rho = obj['rho']        
        Lsum = sum([L * g.edges[eid]['length'] for eid in g.edges()])
        r = np.sqrt(rho * L ** 3 / (np.pi * Lsum))
    else:       
        print('Set relative density to 0.01')
        # rho = 0.01
        rho = 0.15
        Lsum = sum([L * g.edges[eid]['length'] for eid in g.edges()])
        r = np.sqrt(rho * L ** 3 / (np.pi * Lsum))

    for eid in g.edges():
        g.edges[eid]['radius'] = r

    g.graph['rho'] = rho
    space_group = obj.get('space_group', 'unknown')
    Es = obj.get('Es', 0.0)
    g.graph = {'gid':gid, 'cell_type':space_group, 'Es':Es, 'name':'None'}

    for eid in g.edges():        
        g.edges[eid]['Es'] = Es

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

def compute_stats_prop(c_li):
    
    stats_c = {'mean': [], 'std': []}
    E = []
    for c in c_li:
        E.append(c['prop'])    
    stats_c = \
        {
            'mean': np.mean(E),
            'std': np.std(E)            
        }    
    return stats_c

def obj2prop(obj, cid):
    # E = np.array(obj['buckling_strain'])
    E = np.array(obj['E'])
    c = {
        'prop': E,
        'cid': cid,        
    }
    return c

class DatasetCollatedConverter:
    def __init__(self, dn_dataset):
        # Fine-tuning
        dn_ft = os.path.join(dn_dataset, 'Unifying_design_ETH')
        g_dict_ft, c_dict_ft, mapping_pairs_ft = self._get_dataset_ft_wrapper(dn_ft)
        if mix_dataset:
            random.shuffle(mapping_pairs_ft)
        self.split2dataset = \
            self._split_dataset_ft(g_dict_ft, c_dict_ft, mapping_pairs_ft)
            
        # # Curve decoder pre-training
        # self.curves_unlabelled = \
        #     self._get_curves_unlabelled(self.split2dataset['train'])
        
        # # Graph encoder pre-training
        # self.graphs_unlabelled = \
        #     self._get_graphs_unlabelled_wrapper(os.path.join(dn_dataset, 'Pre_training_data/Random_graphs_from_N3_4_D5'))
        # # self.graphs_unlabelled = None
        
        # # Property-prediction pre-training
        # dn_prop = os.path.join(dn_dataset, 'Pre_training_data/Linear_elastic_pre_training')
        # g_dict_prop, c_dict_prop, mapping_pairs_prop = self._get_dataset_graphprop_wrapper(dn_prop)
        # if mix_dataset:
        #     random.shuffle(mapping_pairs_prop)
        # self.split2dataset_prop = \
        #     self._split_dataset_ft(g_dict_prop, c_dict_prop, mapping_pairs_prop)
    
    def _get_dataset_graphprop_wrapper(self, dn_ft):
        g_dict_ft, c_dict_ft, mapping_pairs_ft = {}, {}, []
        offset = {'offset_g':0, 'offset_c':0}
        for fp in os.listdir(dn_ft):
            pn_ft = os.path.join(dn_ft, fp)
            g_dict_subsample, c_dict_subsample, mapping_pairs_subsample = \
                self._get_dataset_graphprop(pn_ft, **offset)
            offset = {'offset_g':len(g_dict_subsample), 'offset_c':len(c_dict_subsample)}
            g_dict_ft.update(g_dict_subsample)
            c_dict_ft.update(c_dict_subsample)
            mapping_pairs_ft.extend(mapping_pairs_subsample)
        x = []
        for c in c_dict_ft.values():
            x.append(c['prop'])
        plt.hist(x)        
        plt.savefig(dn_dataset + '/tmp01.png')

        # collect statistics
        stats_g, stats_c = \
            compute_stats_pretrain(
                g_dict_ft, #split2dataset['train']['g_dict'],
                c_dict_ft, #split2dataset['train']['c_dict'],
            )
        for gid, g in g_dict_ft.items():
            g.graph.update(stats_g)
        for cid, c in c_dict_ft.items():
            c['stats'] = stats_c

        return g_dict_ft, c_dict_ft, mapping_pairs_ft
    
    def _get_dataset_graphprop(self, pn_ft, offset_g=0, offset_c=0, cutoff=0.08):     
        with open(pn_ft) as fp:
            dataset = json.load(fp)
        g_dict = {}
        c_dict = {}
        mapping_pairs = []
        for id, obj in enumerate(dataset):
            gid, cid = id+offset_g, id+offset_c
            g = obj2nxgraph(obj, gid)
            prop = obj2prop(obj, cid)            
            g_dict[gid] = g
            c_dict[cid] = prop
            mapping_pairs.append((gid, cid))
        return g_dict, c_dict, mapping_pairs
    
    
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
        
        c_norm_augmented_2 = []
        # Linear
        for _ in range(10):
            for c_norm in c_norm_augmented:
                c_norm[:,1] = np.linspace(0,1,num = c_norm.shape[0])
                c_norm_augmented_2.append(c_norm)
                print(max(c_norm[:, 1]))
            
        # Peaks-and-valleys
        for _ in range(10*len(curves_unlabelled)):
            c_norm = random.sample(c_norm_augmented, 1)[0]            
            f = np.random.uniform(5,10)
            A = np.random.uniform(0.1,0.5)
            sine_curve = A * np.sin(2*np.pi*f *c_norm[:,0])
            c_norm[:,1] += sine_curve
            c_norm[:, 1] /= max(c_norm[:, 1])
            c_norm_augmented_2.append(c_norm)       
            print(max(c_norm[:, 1]))
        
        # Positive concavity
        for _ in range(10*len(curves_unlabelled)):
            c_norm = random.sample(c_norm_augmented, 1)[0]
            A = np.random.uniform(1,50)
            B = np.random.uniform(1,50)
            exp_curve = A * np.exp(B*c_norm[:,0]) - 1
            c_norm[:,1] *= exp_curve
            c_norm[:,1] /= max(c_norm[:, 1])
            c_norm_augmented_2.append(c_norm)
            print(max(c_norm[:, 1]))
        # for _ in range(10*len(curves_unlabelled)):
        #     c_norm1, c_norm2 = random.sample(c_norm_augmented, 1)
        #     c_norm_new = (c_norm1 + upsample(c_norm2, c_norm1.shape[0])) / 2
        #     c_norm_augmented.append(c_norm_new)
        
        c_norm_augmented.extend(c_norm_augmented_2)
        
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
            offset = len(graphs_unlabelled) # if multiple datasets offset in numbering
        
        # collect statistics
        stats_g = \
            compute_stats_g(graphs_unlabelled                
            )
        for gid, g in enumerate(graphs_unlabelled):
            g.graph.update(stats_g)            
        return graphs_unlabelled

    def _get_graphs_unlabelled(self, pn_graphs, offset=0):        
        # Load single file (dataset)
        with open(pn_graphs) as fp:
            dataset = json.load(fp)
        # Make dict. with 'gid' -> graph
        g_li = []     
        for id, obj in enumerate(dataset):
            gid = id+offset
            g = obj2nxgraph(obj, gid)                        
            g_li.append(g)
        return g_li        

    def _get_dataset_ft_wrapper(self, dn_ft):
        if use_synthetic:
            g_dict_ft, c_dict_ft, mapping_pairs_ft = self._get_dataset_synthetic()
        else:
            g_dict_ft, c_dict_ft, mapping_pairs_ft = {}, {}, []
            offset = {'offset_g':0, 'offset_c':0}
            for fp in os.listdir(dn_ft):
                pn_ft = os.path.join(dn_ft, fp)
                g_dict_subsample, c_dict_subsample, mapping_pairs_subsample = \
                    self._get_dataset_ft(pn_ft, **offset)
                g_dict_ft.update(g_dict_subsample)
                c_dict_ft.update(c_dict_subsample)
                offset = {'offset_g': len(g_dict_ft), 'offset_c':len(c_dict_ft)}
                mapping_pairs_ft.extend(mapping_pairs_subsample)

        # # # curve postprocessing        
        # new_max = min([max(c['curve'][:,0]) for c in c_dict_ft.values()])
        # print('new_max:', new_max)
        # for k, v in list(c_dict_ft.items()):
        #     v['curve'] = set_new_max(v['curve'], new_max)
        #     c_dict_ft[k] = v

        # # y_max_mean = np.array([c['curve'][:, 1].max() for c in c_dict_ft.values()]).mean()
        # for c in c_dict_ft.values():
        #     c['curve'][:, 1] = np.log(50.0*c['curve'][:, 1]/y_max_mean+1.0)
        #     # c['curve'][:, 1] = c['curve'][:, 1]/c['curve'][:, 1].max()

        # y_std = np.stack([c['curve'][:, 1] for c in c_dict_ft.values()], axis=0).std(axis=0)
        # y_mean = np.stack([c['curve'][:, 1] for c in c_dict_ft.values()], axis=0).mean(axis=0)
        # for c in c_dict_ft.values():
        #     x, y = c['curve'][:,0], c['curve'][:,1] #-y_mean)
        #     plt.plot(x, y,linewidth=0.75, alpha=0.8)#, color=f(cy_max))
        # plt.savefig(dn_dataset + '/tmp01.png')

        # # collect statistics
        # stats_g, stats_c = \
        #     compute_stats(
        #         g_dict_ft, #split2dataset['train']['g_dict'],
        #         c_dict_ft, #split2dataset['train']['c_dict'],
        #     )
        # for gid, g in g_dict_ft.items():
        #     g.graph.update(stats_g)
        # for cid, c in c_dict_ft.items():
        #     c['stats'] = stats_c

        return g_dict_ft, c_dict_ft, mapping_pairs_ft

    def _get_dataset_ft(self, pn_ft, offset_g=0, offset_c=0, cutoff=0.08):
        from src.dataset_preprocessing_collated import obj2curve_ETH
        with open(pn_ft) as fp:
            dataset = json.load(fp)
        g_dict = {}
        c_dict = {}
        mapping_pairs = []
        for id, obj in enumerate(tqdm(dataset)):
            if id < 198:
                continue
            gid, cid = id+offset_g, id+offset_c
            g = obj2nxgraph(obj, gid)
            # g = tesselate(g, base='octet')
            curve = obj2curve_ETH(obj, cid)
            if not use_synthetic:
                try:
                    _ = untesselate(g)
                except:
                    print('cannot tesselate')
                    continue

            if curve['curve'][:, 1].max() <= 0.0:
                print('malformatted curve')
                continue

            if skip_non_monotonic and np.sum(curve['curve'][1:] < curve['curve'][:-1]) != 0:
                continue
            g_dict[gid] = g
            c_dict[cid] = curve
            mapping_pairs.append((gid, cid))
            # if len(g_dict) > 50_000:
            #     break
        # print(list(g_dict.keys()))
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
            print(split, len(c_dict_split))                   
            split2dataset[split] = \
                {
                    'g_dict': g_dict_split,
                    'c_dict': c_dict_split,
                    'mapping_pairs': mapping_pairs_split
                }

        return split2dataset

def main():
    assert dn_out != dn_dataset
    # os.system(f'mkdir {dn_out}')
    os.mkdir(dn_out)

    print('Data processing start...')
    dataset_converter = DatasetCollatedConverter(dn_dataset)
    split2dataset = dataset_converter.split2dataset
    # split2dataset_prop = dataset_converter.split2dataset_prop
    # curves_unlabelled = dataset_converter.curves_unlabelled
    # graphs_unlabelled = dataset_converter.graphs_unlabelled

    print('Data saving start...')
    print('Fine-tuning data saving start...')
    for split, data_dict in split2dataset.items():
        save_dataset_ft(**data_dict, dn_out=dn_out, split=split)
    
    print('Pre-training data saving start...')
    # save_dataset_pt(curves_unlabelled, graphs_unlabelled, dn_out)
    # save_dataset_pt(curves_unlabelled, [], dn_out)
    # for split, data_dict in split2dataset_prop.items():
    #     save_dataset_ft(**data_dict, dn_out=dn_out, split=split)
    print('Data saving done.')

def save_dataset_ft(g_dict, c_dict, mapping_pairs, dn_out, split):
    pn_match = os.path.join(dn_out, split)
    pn_graphs = os.path.join(dn_out, split, "graphs")
    pn_curves = os.path.join(dn_out, split, "curves")
    # os.system(f'mkdir {pn_match}')
    # os.system(f'mkdir {pn_graphs}')
    # os.system(f'mkdir {pn_curves}')
    os.mkdir(pn_match)
    os.mkdir(pn_graphs)
    os.mkdir(pn_curves)

    # Save curves
    for cid, c in c_dict.items():
        with open(os.path.join(pn_curves,f'{cid}.pkl'), 'wb') as fp:
            pickle.dump(c, fp)

    # save graphs
    for gid, g in g_dict.items():
        # g = rm_redundant_nodes(g)
        with open(os.path.join(pn_graphs, f'{gid}.gpkl'), 'wb') as f:
            pickle.dump(g, f)

        # for nid in g.nodes():
        #     g.nodes[nid]['coord'] = g.nodes[nid]['coord']

        if not use_synthetic:
            g2 = untesselate(g)
            with open(os.path.join(pn_graphs, f'{gid}_polyhedron.gpkl'), 'wb') as f:
                pickle.dump(g2, f)

        # g3 = tesselate(g2.copy())

        # for nid in g3.nodes():
        #     g3.nodes[nid]['coord'] = 2.0*(g3.nodes[nid]['coord'] - 0.5)
        #
        # plot_3d(g1, os.path.join(pn_graphs, f'tesselated_{gid}.png'))
        # plot_3d(g3, os.path.join(pn_graphs, f'tesselated_{gid}_again.png'))
        # plot_3d(g2, os.path.join(pn_graphs, f'untesselated_{gid}.png'))

    # nx.write_gpickle(g, os.path.join(pn_graphs,f'{gid}.gpkl'))
    # save mapping
    with open(os.path.join(pn_match,'mapping.tsv'), 'w+') as fp:
        fp.writelines(['\t'.join([str(id) for id in mapping])+'\n' for mapping in mapping_pairs])


def save_dataset_pt(c_li, g_li, dn_out):
    # pn_graphs_unlabelled = os.path.join(dn_out, "unlabelled_graphs")
    # os.system(f'mkdir {pn_graphs_unlabelled}')
    pn_curves_unlabelled = os.path.join(dn_out, "unlabelled_curves")
    # os.system(f'mkdir {pn_curves_unlabelled}')
    os.mkdir(pn_curves_unlabelled)
    # os.mkdir(pn_graphs_unlabelled)

    # Save curves
    for cid, c in enumerate(c_li):
        with open(os.path.join(pn_curves_unlabelled,f'{cid}.pkl'), 'wb') as fp:
            pickle.dump(c, fp)
    
    # # Save graphs
    # for gid in range(len(g_li)):
    #     g = g_li[gid]
    #     with open(os.path.join(pn_graphs_unlabelled,f'{gid}.pkl'), 'wb') as fp:
    #         pickle.dump(g, fp)

if __name__ == '__main__':
    main()
    # augment()

