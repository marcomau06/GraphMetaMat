import torch
import time
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from tslearn.clustering import TimeSeriesKMeans

from src.config import args
from src.dataset import DataLoaderFactory
from src.dataset_curve import get_curve

N_BINS_MAGNITUDE = 10
N_BINS_SHAPE = 50
SUBSAMPLE = None
USE_ZSCORE = True

def main():
    ######## Seed for reproducible results ##########
    import random
    import numpy as np
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    ############# Datasets loading ####################
    args['dataset'].update(args['dataset_RL'])
    dlf_rl = DataLoaderFactory(**args['dataset'])

    for split in [dlf_rl.train_split, dlf_rl.valid_split, dlf_rl.test_split]:
        curve_split_pn = os.path.join(dlf_rl.root_curve, split, 'curves')
        binning_split_pn = os.path.join(dlf_rl.root_curve, split, 'binning')
        print(f'Saving to {binning_split_pn}...')
        os.system(f'mkdir {binning_split_pn}') # TODO: Process for Windows Version

        # get train dataset
        cid_li = []
        X_train_magnitude = []
        X_train_shape = []
        for fn in tqdm(os.listdir(curve_split_pn)):
            cid = int(fn.split('.')[0])
            cid_li.append(cid)

            curve_pn = os.path.join(curve_split_pn, fn)
            with open(curve_pn, 'rb') as fp:
                c = pickle.load(fp)
            c_magnitude, c_shape = get_curve(c['curve'], **dlf_rl.curve_norm_cfg)
            L, M = c_shape.shape
            assert M == 1
            X_train_magnitude.append(float(c_magnitude))
            X_train_shape.append(c_shape.reshape(L).detach().cpu().numpy())
        X_train_shape = np.stack(X_train_shape, axis=0)
        X_train_shape_mean = np.expand_dims(np.mean(X_train_shape, axis=0), axis=0)
        if USE_ZSCORE:
            X_train_shape = X_train_shape - X_train_shape_mean
        if SUBSAMPLE is not None:
            X_train_shape = X_train_shape[random.sample(list(range(X_train_shape.shape[0])), k=SUBSAMPLE)]
        X_train_magnitude = torch.tensor(X_train_magnitude)
        labels_magnitude = \
            torch.bucketize(
                X_train_magnitude,
                torch.arange(
                    X_train_magnitude.max() - X_train_magnitude.min()
                )/N_BINS_MAGNITUDE + X_train_magnitude.min()).detach().cpu().numpy()

        # fit clustering algorithm
        kmeans = TimeSeriesKMeans(n_clusters=N_BINS_SHAPE, metric="dtw", max_iter=10, random_state=1)
        print('running KMeans...')
        kmeans.fit(X_train_shape)
        print('KMeans finished!')
        labels_shape = kmeans.labels_
        centers = kmeans.cluster_centers_

        binning_dict = {
            'data_shape': {
                cid: {'label': label, 'count': sum(label == labels_shape)}
                for cid, label in zip(cid_li, labels_shape)
            },
            'data_magnitude': {
                cid: {'label': label, 'count': sum(label == labels_magnitude)}
                for cid, label in zip(cid_li, labels_magnitude)
            },
            'num_classes_shape': N_BINS_SHAPE,
            'num_classes_magnitude': N_BINS_MAGNITUDE,
            'zscore': USE_ZSCORE
        }

        # plot centers
        print('plotting...')
        for lid in range(N_BINS_SHAPE):
            mask = (labels_shape == lid)
            if True not in mask:
                continue

            if USE_ZSCORE:
                best_curve = centers[lid].reshape(-1) + X_train_shape_mean.reshape(-1)
                train_curve_li = X_train_shape[mask] + X_train_shape_mean
            else:
                best_curve = centers[lid].reshape(-1)
                train_curve_li = X_train_shape[mask]
            plt.clf()
            for train_curve in train_curve_li:
                plt.plot(train_curve, color='black', linewidth=1, alpha=0.5)
            plt.plot(best_curve, color='red', linewidth=2, alpha=1.0)
            plt.savefig(os.path.join(binning_split_pn, f'viz-{lid}.png'))
            plt.close()

        with open(os.path.join(binning_split_pn, f'binning_dict.pkl'), 'wb') as fp:
            pickle.dump(binning_dict, fp)



if __name__ == '__main__':
    main()