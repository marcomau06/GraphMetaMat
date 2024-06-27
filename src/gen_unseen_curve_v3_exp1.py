# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:42:42 2024

@author: Marco Maurizi
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np
import os

from tqdm import tqdm
from scipy.interpolate import CubicSpline

import random
import json
import pickle

wDr = os.getcwd()

# %%
def load_dataset(dn_ft):
    dataset = []
    for fp in os.listdir(dn_ft):
        print(fp)
        pn_ft = os.path.join(dn_ft, fp)

        with open(pn_ft) as json_file:
            tmp = json.load(json_file)

        dataset.extend(tmp)
    return dataset

# %%
dn = '/home/derek/Documents/MaterialSynthesis/data/marco_02142024_PSU'
dataset = load_dataset(os.path.join(dn, 'Fine_tuning_data'))

def main_v5():
    fn = 'oodv5_exp3_v3'

    resolution = 256

    start = 0.0
    end = -30.0
    ybase = start + (end-start) * np.arange(resolution) / resolution
    x = np.arange(resolution)

    window = 20
    curve_li_all = []
    for i in range(window, resolution-window):
        # ys = [start, ybase[i]+20.0, end]
        # xs = [0.0, i, resolution]
        # ys = [start, ybase[i-int(window/2)]+10, ybase[i]+20.0, ybase[i+int(window/2)]+10, end]
        # xs = [0.0, i-int(window/2), i, i+int(window/2), resolution]
        mp1, mp2 = int(i/2), int(i+(resolution-i)/2)
        ys = [start, ybase[mp1], ybase[i]-20.0, ybase[mp2], end]
        xs = [0.0, mp1, i, mp2, resolution]
        cs = CubicSpline(xs, ys)
        curve_li_all.append(cs(x))
    num_samples = len(curve_li_all)

    indices_to_sample = list(range(num_samples))[5:-5:10] #[::10]
    curve_li_train = [curve_li_all[x] for x in range(num_samples) if x not in indices_to_sample]
    curve_li_dev = [curve_li_all[x] for x in indices_to_sample][::2]
    curve_li_test = [curve_li_all[x] for x in indices_to_sample][1::2]

    random.seed(123)
    random.shuffle(curve_li_train)

    id = 0
    os.system(f"rm -rf {os.path.join(dn, fn)}")
    os.system(f"mkdir {os.path.join(dn, fn)}")
    for split, curve_li in [
        ('train', curve_li_train),
        ('dev', curve_li_dev),
        ('test', curve_li_test),
    ]:
        id_li = []
        os.system(f"mkdir {os.path.join(dn, fn, split)}")
        os.system(f"mkdir {os.path.join(dn, fn, split, 'graphs')}")
        os.system(f"mkdir {os.path.join(dn, fn, split, 'curves')}")
        for c_curve in tqdm(curve_li):
            pn_in1 = os.path.join(dn, 'preprocessed_unitcell_True_mixed', 'train', 'graphs', '0.gpkl')
            pn_in2 = os.path.join(dn, 'preprocessed_unitcell_True_mixed', 'train', 'graphs', '0_polyhedron.gpkl')
            pn_out1 = os.path.join(dn, fn, split, 'graphs', f'{id}.gpkl')
            pn_out2 = os.path.join(dn, fn, split, 'graphs', f'{id}_polyhedron.gpkl')
            id_li.append(id)

            with open(pn_in1, 'rb') as f:
                g = pickle.load(f)
            g.graph['gid'] = id
            with open(pn_out1, 'wb') as f:
                pickle.dump(g, f)
            # os.system(f'cp {pn_in1} {pn_out1}')
            os.system(f'cp {pn_in2} {pn_out2}')

            curve = {
                'curve': np.stack((np.arange(len(c_curve)) / (len(c_curve) - 1), c_curve), axis=1),
                'cid': id, 'is_monotonic': True
            }
            with open(os.path.join(dn, fn, split, 'curves', f'{id}.pkl'), 'wb') as fp:
                pickle.dump(curve, fp)
            id += 1

        with open(os.path.join(dn, fn, split, 'mapping.tsv'), 'w+') as fp:
            fp.writelines([f'{id}\t{id}\n' for id in id_li])

def main_v7():
    double_splits = True
    # fn = 'ood_positive_sliding_window128'
    fn = 'ood_negative_sliding_window16_double3'
    resolution = 256
    # resolution_ds = 128
    resolution_ds = 16
    if double_splits:
        window_size = int(3/16*resolution_ds)
    else:
        window_size = int(4/16*resolution_ds)

    curve_li_all = []
    assert resolution % resolution_ds == 0
    if double_splits:
        for i in range(resolution_ds-window_size):
            for j in range(i+window_size*2, resolution_ds-window_size):
                if 'positive' in fn:
                    curve = np.zeros(resolution_ds)
                    curve[i:i+window_size] = 1
                    curve[j:j+window_size] = 1
                elif 'negative' in fn:
                    curve = np.ones(resolution_ds)
                    curve[i:i+window_size] = 0
                    curve[j:j+window_size] = 0
                else:
                    assert False

                curve = 60.0 * curve - 40.0
                curve = np.repeat(curve, int(resolution / resolution_ds))
                curve_li_all.append(curve)
    else:
        for i in range(resolution_ds-window_size):
            if 'positive' in fn:
                curve = np.zeros(resolution_ds)
                curve[i:i+window_size] = 1
            elif 'negative' in fn:
                curve = np.ones(resolution_ds)
                curve[i:i+window_size] = 0
            else:
                assert False

            curve = 60.0*curve - 40.0
            curve = np.repeat(curve, int(resolution/resolution_ds))
            curve_li_all.append(curve)

    curve_li_train = curve_li_all
    curve_li_dev = curve_li_all
    curve_li_test = curve_li_all
    # num_samples = len(curve_li_all)
    # indices_to_sample = list(range(num_samples))[2:-2:10] #[::10]
    # curve_li_train = [curve_li_all[x] for x in range(num_samples) if x not in indices_to_sample]
    # curve_li_dev = [curve_li_all[x] for x in indices_to_sample][::2]
    # curve_li_test = [curve_li_all[x] for x in indices_to_sample][1::2]

    id = 0
    os.system(f"rm -rf {os.path.join(dn, fn)}")
    os.system(f"mkdir {os.path.join(dn, fn)}")
    for split, curve_li in [
        ('train', curve_li_train),
        ('dev', curve_li_dev),
        ('test', curve_li_test),
    ]:
        id_li = []
        os.system(f"mkdir {os.path.join(dn, fn, split)}")
        os.system(f"mkdir {os.path.join(dn, fn, split, 'graphs')}")
        os.system(f"mkdir {os.path.join(dn, fn, split, 'curves')}")
        for c_curve in tqdm(curve_li):
            pn_in1 = os.path.join(dn, 'preprocessed_unitcell_True_mixed', 'train', 'graphs', '0.gpkl')
            pn_in2 = os.path.join(dn, 'preprocessed_unitcell_True_mixed', 'train', 'graphs', '0_polyhedron.gpkl')
            pn_out1 = os.path.join(dn, fn, split, 'graphs', f'{id}.gpkl')
            pn_out2 = os.path.join(dn, fn, split, 'graphs', f'{id}_polyhedron.gpkl')
            id_li.append(id)

            with open(pn_in1, 'rb') as f:
                g = pickle.load(f)
            g.graph['gid'] = id
            with open(pn_out1, 'wb') as f:
                pickle.dump(g, f)
            # os.system(f'cp {pn_in1} {pn_out1}')
            os.system(f'cp {pn_in2} {pn_out2}')

            curve = {
                'curve': np.stack((np.arange(len(c_curve)) / (len(c_curve) - 1), c_curve), axis=1),
                'cid': id, 'is_monotonic': True
            }
            with open(os.path.join(dn, fn, split, 'curves', f'{id}.pkl'), 'wb') as fp:
                pickle.dump(curve, fp)
            id += 1

        with open(os.path.join(dn, fn, split, 'mapping.tsv'), 'w+') as fp:
            fp.writelines([f'{id}\t{id}\n' for id in id_li])

def main_v3():
    fn = 'oodv3_exp1_v6_flat'

    num_samples = 1000
    resolution = 256

    lower_bound = -40.0
    upper_bound = 5.0 #-20.0 #10.0
    alpha = 0.0
    gamma = 0.1
    epsilon = 0.1
    curve_li_all = []
    for i in range(num_samples):
        beta = lower_bound + (upper_bound - lower_bound) * i / (num_samples - 1)
        x = np.arange(resolution)/(resolution-1)
        # get curve
        # alpha = beta + 10.0
        # curve = (alpha - beta) * np.exp(np.log(epsilon)/gamma * x) + beta
        curve = beta * np.ones_like(x)
        curve_li_all.append(curve)

    indices_to_sample = list(range(num_samples))[5:-5:10] #[::10]
    curve_li_train = [curve_li_all[x] for x in range(num_samples) if x not in indices_to_sample]
    curve_li_dev = [curve_li_all[x] for x in indices_to_sample][::2]
    curve_li_test = [curve_li_all[x] for x in indices_to_sample][1::2]

    random.seed(123)
    random.shuffle(curve_li_train)

    id = 0
    os.system(f"rm -rf {os.path.join(dn, fn)}")
    os.system(f"mkdir {os.path.join(dn, fn)}")
    for split, curve_li in [
        ('train', curve_li_train),
        ('dev', curve_li_dev),
        ('test', curve_li_test),
    ]:
        id_li = []
        os.system(f"mkdir {os.path.join(dn, fn, split)}")
        os.system(f"mkdir {os.path.join(dn, fn, split, 'graphs')}")
        os.system(f"mkdir {os.path.join(dn, fn, split, 'curves')}")
        for c_curve in tqdm(curve_li):
            pn_in1 = os.path.join(dn, 'preprocessed_unitcell_True_mixed', 'train', 'graphs', '0.gpkl')
            pn_in2 = os.path.join(dn, 'preprocessed_unitcell_True_mixed', 'train', 'graphs', '0_polyhedron.gpkl')
            pn_out1 = os.path.join(dn, fn, split, 'graphs', f'{id}.gpkl')
            pn_out2 = os.path.join(dn, fn, split, 'graphs', f'{id}_polyhedron.gpkl')
            id_li.append(id)

            with open(pn_in1, 'rb') as f:
                g = pickle.load(f)
            g.graph['gid'] = id
            with open(pn_out1, 'wb') as f:
                pickle.dump(g, f)
            # os.system(f'cp {pn_in1} {pn_out1}')
            os.system(f'cp {pn_in2} {pn_out2}')

            curve = {
                'curve': np.stack((np.arange(len(c_curve)) / (len(c_curve) - 1), c_curve), axis=1),
                'cid': id, 'is_monotonic': True
            }
            with open(os.path.join(dn, fn, split, 'curves', f'{id}.pkl'), 'wb') as fp:
                pickle.dump(curve, fp)
            id += 1

        with open(os.path.join(dn, fn, split, 'mapping.tsv'), 'w+') as fp:
            fp.writelines([f'{id}\t{id}\n' for id in id_li])

if __name__ == '__main__':
    main_v7()