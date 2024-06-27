# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:42:42 2024

@author: Marco Maurizi
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from tqdm import tqdm
import numpy as np
import os

import json
import pickle

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

def resample_curve(x, y, S):
    # Calculate interval width
    interval_width = (max(x) - min(x)) / S

    # Preallocate arrays to store resampled x and y values
    resampled_x = np.empty(S)
    resampled_y = np.empty(S)

    # Binning x values into intervals
    bins = np.digitize(x, np.linspace(min(x), max(x), S + 1))

    # Iterate through each interval
    for i in range(1, S + 1):
        # Find indices of x within the current interval
        indices = np.where(bins == i)[0]

        # Calculate average y value within the interval
        if len(indices) > 0:
            avg_y = np.mean(y[indices])

        # Store resampled values
        interval_start = min(x) + (i - 1) * interval_width
        interval_end = interval_start + interval_width
        resampled_x[i - 1] = (interval_start + interval_end) / 2
        resampled_y[i - 1] = avg_y

    return resampled_x, resampled_y


def vector_to_binary(vector, threshold):
    binary_vector = (vector >= threshold).astype(int)
    return binary_vector

# plt.plot(x,y)
# plt.plot(resampled_x,resampled_y)

# %% Binary sequences with S bits
from itertools import product

def generate_binary_sequences(S):
    # Generate all possible combinations of binary sequences of length S
    sequences = list(product([0, 1], repeat=S))
    return sequences

def gen_unseen_seq(dn, S, shuffle_on, dist_limit, un_curves):
    dataset = load_dataset(dn)

    # %% Dataset resampling and binarization
    dataset_binary = []
    for cell in dataset:
        x = np.array(cell['frequency'])
        y = np.array(cell['transmissibility'])
        resampled_x, resampled_y = resample_curve(x, y, S)

        y_binary = vector_to_binary(resampled_y, threshold=-30)
        dataset_binary.append(y_binary)

    all_seq = generate_binary_sequences(S)

    if shuffle_on:
        import random
        random.shuffle(all_seq)

    # %% Generate unseen curves
    unseen_seq = []
    un_curves = min(un_curves, len(all_seq))
    print('un_curves', un_curves)
    for seq_id in tqdm(range(un_curves)):
        seq = np.array(all_seq[seq_id])

        for count, data_binary in enumerate(dataset_binary):
            dist = np.sum(seq != data_binary) / len(seq)  # 1 - accuracy
            if dist < dist_limit:
                break
            if count == len(dataset_binary) - 1:
                unseen_seq.append(seq.astype(float)*60.0-40.0)

    return unseen_seq

if __name__ == '__main__':
    # %%
    S = 16  # Number of frequency intervals
    shuffle_on = True #False  # shuffle all sequences
    dist_limit = 0.35 #25  # limit in 1-accuracy
    un_curves = 100000  # number of tentative sequences
    dn = '/home/derek/Documents/MaterialSynthesis/data/marco_02142024_PSU'
    out = gen_unseen_seq(os.path.join(dn, 'Fine_tuning_data'), S, shuffle_on, dist_limit, un_curves)
    print(out)

    id = 0
    os.system(f"rm -rf {os.path.join(dn, 'ood_data')}")
    os.system(f"mkdir {os.path.join(dn, 'ood_data')}")
    for split, c_curve_li in zip(['train', 'dev', 'test'], [out[:-100], out[-100:-50], out[-50:]]):
        os.system(f"mkdir {os.path.join(dn, 'ood_data', split)}")
        os.system(f"mkdir {os.path.join(dn, 'ood_data', split, 'graphs')}")
        os.system(f"mkdir {os.path.join(dn, 'ood_data', split, 'curves')}")

        id_li = []
        for c_curve in tqdm(c_curve_li):
            pn_in1 = os.path.join(dn, 'preprocessed_unitcell_True_mixed', 'train', 'graphs', '0.gpkl')
            pn_in2 = os.path.join(dn, 'preprocessed_unitcell_True_mixed', 'train', 'graphs', '0_polyhedron.gpkl')
            pn_out1 = os.path.join(dn, 'ood_data', split, 'graphs', f'{id}.gpkl')
            pn_out2 = os.path.join(dn, 'ood_data', split, 'graphs', f'{id}_polyhedron.gpkl')
            id_li.append(id)

            with open(pn_in1, 'rb') as f:
                g = pickle.load(f)
            g.graph['gid'] = id
            with open(pn_out1, 'wb') as f:
                pickle.dump(g, f)
            # os.system(f'cp {pn_in1} {pn_out1}')
            os.system(f'cp {pn_in2} {pn_out2}')

            curve = {
                'curve': np.stack((np.arange(len(c_curve))/(len(c_curve)-1), c_curve), axis=1),
                'cid': id, 'is_monotonic': True
            }
            with open(os.path.join(dn, 'ood_data', split, 'curves', f'{id}.pkl'), 'wb') as fp:
                pickle.dump(curve, fp)
            id += 1

        with open(os.path.join(dn, 'ood_data', split, 'mapping.tsv'), 'w+') as fp:
            fp.writelines([f'{id}\t{id}\n' for id in id_li])


    print(len(out))#, out)