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

# %%#######################################################
S = 16  # Number of frequency intervals


##########################################################

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


def vector_to_binary(vector, threshold=-20):
    binary_vector = (vector >= threshold).astype(int)
    return binary_vector


def hamming_dist_norm(seq1, seq2):
    assert len(seq1) == len(seq2)
    return np.sum(seq1 != seq2) / len(seq1)


def _distance_matrix(seqs1, seqs2):
    distance_matrix = np.zeros((len(seqs1), len(seqs2)))
    for i in range(len(seqs1)):
        for j in range(len(seqs2)):
            distance_matrix[i, j] = hamming_dist_norm(seqs1[i], seqs2[j])
    return distance_matrix


# %%
dn = '/home/derek/Documents/MaterialSynthesis/data/marco_02142024_PSU'
dataset = load_dataset(os.path.join(dn, 'Fine_tuning_data'))

# %% Dataset resampling and binarization
dataset_binary = []
for cell in dataset:
    x = np.array(cell['frequency'])
    y = np.array(cell['transmissibility'])
    resampled_x, resampled_y = resample_curve(x, y, S)

    # y_binary = vector_to_binary(resampled_y, threshold = -20)
    # dataset_binary.append(y_binary)
    break

cfg_li = [
    ('binary', [[0,4000]]),
    ('random', [[0,4000]]),
    ('binary', [[3000, 5000], [8000, 10000]]),
    ('random', [[3000, 5000], [8000, 10000]]),
    ('binary', [[0, 4000], [8000, 10000]]),
    ('random', [[0, 4000], [8000, 10000]]),
]


for init_method, freq_lims in cfg_li:
    fn = f'{init_method}-{"-".join([str(lb) + "_" + str(ub) for (lb, ub) in freq_lims])}'

    os.system(f"mkdir {os.path.join(dn, fn)}")

    c_curve_li = []
    if init_method == 'binary':
        seq = np.ones((S,))
        for x_t in freq_lims:
            l_id = np.argmin(abs(resampled_x - x_t[0]))
            u_id = np.argmin(abs(resampled_x - x_t[1]))

            seq[l_id:u_id] = 0
        c_curve_li.append(seq.astype(float) * 60.0 - 40.0)
    elif init_method == 'random':
        NUM_RANDOM_INIT = 20
        for _ in range(NUM_RANDOM_INIT):
            seq = np.random.randint(2, size=(S,))
            for x_t in freq_lims:
                l_id = np.argmin(abs(resampled_x - x_t[0]))
                u_id = np.argmin(abs(resampled_x - x_t[1]))

                seq[l_id:u_id] = 0
            c_curve_li.append(seq.astype(float) * 60.0 - 40.0)
    else:
        assert False

    id = 0
    id_li = []
    split = 'test'
    os.system(f"mkdir {os.path.join(dn, fn, split)}")
    os.system(f"mkdir {os.path.join(dn, fn, split, 'graphs')}")
    os.system(f"mkdir {os.path.join(dn, fn, split, 'curves')}")
    for c_curve in tqdm(c_curve_li):
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

    # seq = np.zeros((S,), dtype = np.int32)
    # seq = np.ones((S,), dtype = np.int32)



# # %% Target binary sequences with S bits - 2
# freq_lims = [[3000, 5000], [8000, 10000]]
# seq = np.random.randint(2, size=(S,))
# # seq = np.zeros((S,), dtype = np.int32)
# # seq = np.ones((S,), dtype = np.int32)
# for x_t in freq_lims:
#     l_id = np.argmin(abs(resampled_x - x_t[0]))
#     u_id = np.argmin(abs(resampled_x - x_t[1]))
#
#     seq[l_id:u_id] = 0
#
# print(seq)
