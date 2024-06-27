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


import random
if __name__ == '__main__':
    fn = 'oodv3_exp2_v7'
    resolution = 128
    band_width = int(resolution / 8)

    curve_li_all = []
    for i in range(int(0.4 * resolution) - band_width):
        # curve = np.zeros((resolution,))
        # curve[i:i + band_width] = 1.0
        curve = np.ones((resolution,))
        curve[i:i + band_width] = 0.0
        curve = curve.astype(float) * 60.0 - 40.0
        curve_li_all.append(curve)

    num_samples = len(curve_li_all)
    indices_to_sample = list(range(num_samples))[::5]
    curve_li_train = [curve_li_all[x] for x in range(num_samples) if x not in indices_to_sample]
    curve_li_dev = [curve_li_all[x] for x in indices_to_sample][::2]
    curve_li_test = [curve_li_all[x] for x in indices_to_sample][1::2]

    # random.seed(123)
    # random.shuffle(curve_li_all)

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
