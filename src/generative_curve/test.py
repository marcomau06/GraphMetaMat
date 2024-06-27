import torch
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy

from src.config import ETH_FULL_C_VECTOR
from src.utils import get_shape
from src.utils import get_mse, get_mae, get_jaccard_index_monotonic, get_mape, to_numpy
from scipy import stats

from src.utils import plot_3d, plot_magnitude, plot_curve, CObj, r2_score

def plot_pred(graph_li, curve_pred_li, curve_true_li, skip_magnitude=False, dn=None, max_figs=500):
    magnitude_pred_all = np.array([curve_pred.c_magnitude for curve_pred in curve_pred_li])
    magnitude_true_all = np.array([curve_true.c_magnitude for curve_true in curve_true_li])
    for i, (graph, curve_pred, curve_true) in \
            enumerate(tqdm(zip(graph_li, curve_pred_li, curve_true_li))):
        fig = plt.figure(figsize=(20,12))
        ax = fig.add_subplot(221, projection="3d")
        # plt.subplot(2,2,1)
        ax.set_title('Graph')
        plot_3d(graph, ax=ax)
        ax = fig.add_subplot(222)
        # plt.subplot(2,2,2)
        plt.gca().set_title('Curve')
        plot_curve({
            'pred':(curve_pred.c,
                    curve_pred.c_UBLB[0] if curve_pred.c_UBLB is not None else None,
                    curve_pred.c_UBLB[1] if curve_pred.c_UBLB is not None else None),
            'true':(curve_true.c,
                    curve_true.c_UBLB[0] if curve_true.c_UBLB is not None else None,
                    curve_true.c_UBLB[1] if curve_true.c_UBLB is not None else None),
        }, ax=ax)
        ax = fig.add_subplot(223)
        # plt.subplot(2,2,3)
        if not skip_magnitude:
            plt.gca().set_title('Magnitude')
            plot_magnitude(magnitude_pred_all, magnitude_true_all, i, ax=ax)
        # plt.subplot(2,2,4)
        ax = fig.add_subplot(224)
        plt.gca().set_title('Curve (shape, z-score)')

        c_shape_pred_UBLB = \
            (
                curve_pred.c_shape + curve_pred.c_shape_std,
                curve_pred.c_shape - curve_pred.c_shape_std
            ) if curve_pred.c_UBLB is not None else None
        plot_curve({
            'pred (z-score shape)':(curve_pred.c_shape,
                                    c_shape_pred_UBLB[0] if c_shape_pred_UBLB is not None else None,
                                    c_shape_pred_UBLB[1] if c_shape_pred_UBLB is not None else None),
            'true (z-score shape)':(curve_true.c_shape, None, None)
        }, ax=ax)
        fig.tight_layout()
        plt.savefig(os.path.join(dn, f'{i}.png'))
        plt.clf()
        plt.close()
        if i > max_figs:
            break

def get_r2_score(t, p):
    try:
        r_squared = r2_score(t, p)
    except:
        from pprint import pprint
        print('r2 values too large')
        r_squared = float('inf')
    return r_squared

def test(dataset, model, device, plot_num_samples_max=None):
    dataset_train = dataset
    model.eval()
    model = model.to(device)
    metrics_raw = defaultdict(list)
    graph_li, curve_pred_li, curve_true_li = [], [], []
    t_magnitude_li, p_magnitude_li = [], []
    t_magnitude_norm_li, p_magnitude_norm_li = [], []

    mmmt = []
    mmmp = []
    for i, (graph, curve_true) in enumerate(tqdm(dataset)):
        graph.to(device), curve_true.to(device)
        c_shape_true, c_magnitude_true = curve_true.c_shape, curve_true.c_magnitude
        if ETH_FULL_C_VECTOR:
            c_shape_pred, c_magnitude_pred, c_shape_pred_std, c_magnitude_pred_std = \
                model.inference(graph, return_cn=False)
            for i in range(len(graph.g_li)):
                graph.g_li[i].graph['C'] = 1.0
                graph.g_li[i].graph['n'] = 1.0
        else:
            c_shape_pred, c_magnitude_pred, c_shape_pred_std, c_magnitude_pred_std, cn = \
                model.inference(graph, return_cn=True)
            for i, cn_elt in enumerate(cn):
                C, n = cn_elt
                graph.g_li[i].graph['C'] = C.item()
                graph.g_li[i].graph['n'] = n.item()

        c_pred, c_pred_UBLB = \
            dataset_train.dataset.unnormalize_curve(
                c_magnitude_pred, c_shape_pred,
                c_magnitude_u=c_magnitude_pred_std,
                c_shape_u=c_shape_pred_std)
        c_true, _ = \
            dataset_train.dataset.unnormalize_curve(
                c_magnitude_true, c_shape_true)

        if ETH_FULL_C_VECTOR:
            mmmt.append(c_magnitude_true.detach().cpu().numpy())
            mmmp.append(c_magnitude_pred.detach().cpu().numpy())

        t_magnitude_li.append(
            (
                c_true.detach().cpu().numpy().max(axis=1).flatten() -
                c_true.detach().cpu().numpy().min(axis=1).flatten()
            ).flatten())
        p_magnitude_li.append(
            (
                c_pred.detach().cpu().numpy().max(axis=1).flatten() -
                c_pred.detach().cpu().numpy().min(axis=1).flatten()
            ).flatten())
        t_magnitude_norm_li.append(
            c_magnitude_true.detach().cpu().numpy().flatten())
        p_magnitude_norm_li.append(
            c_magnitude_pred.detach().cpu().numpy().flatten())

        metrics_single_batch = get_metrics(c_pred, c_true, True) # is_monotonic

        for k,v in metrics_single_batch.items():
            metrics_raw[k].extend(v)

        # update graph_li and c_li
        if plot_num_samples_max is None or len(graph_li) < plot_num_samples_max:
            graph_li.extend(graph.g_li)
            if ETH_FULL_C_VECTOR:
                continue
            for i in range(len(c_pred)):
                curve_pred_li.append(CObj(
                    c=c_pred[i].view(-1).detach().cpu().numpy(),
                    c_magnitude=float(c_magnitude_pred[i]),
                    c_shape=c_shape_pred[i].view(-1).detach().cpu().numpy(),
                    c_magnitude_std=c_magnitude_pred_std[i].view(-1).detach().cpu().numpy() \
                        if c_magnitude_pred_std is not None else None,
                    c_shape_std=c_shape_pred_std[i].view(-1).detach().cpu().numpy() \
                        if c_shape_pred_std is not None else None,
                    c_UBLB=(
                        c_pred_UBLB[0][i].view(-1).detach().cpu().numpy(),
                        c_pred_UBLB[1][i].view(-1).detach().cpu().numpy()) \
                        if c_pred_UBLB is not None else None
                ))
                if c_pred_UBLB is not None and c_pred_UBLB[0][i].detach().cpu().numpy().reshape(-1).shape[0] > 256:
                    print('asdf')

                curve_true_li.append(CObj(
                    c=c_true[i].view(-1).detach().cpu().numpy(),
                    c_magnitude=float(c_magnitude_true[i]),
                    c_shape=c_shape_true[i].view(-1).detach().cpu().numpy()
                ))

    # export metrics
    metrics_collated = collate_metrics(metrics_raw)

    if ETH_FULL_C_VECTOR:
        mmmt = np.concatenate(mmmt, axis=0)
        mmmp = np.concatenate(mmmp, axis=0)
        r2 = 0
        for i in range(mmmt.shape[-1]):
            zzz = get_r2_score(mmmt[:,i], mmmp[:,i])
            r2 = r2 + zzz
            metrics_collated[f'r2_{i}'] = [zzz, 0]
        r2 = r2/mmmt.shape[-1]
    else:
        r2 = \
            get_r2_score(
                np.concatenate(t_magnitude_li),
                np.concatenate(p_magnitude_li))
        r2_zscore = \
            get_r2_score(
                np.concatenate(t_magnitude_norm_li),
                np.concatenate(p_magnitude_norm_li))
        metrics_collated['r2_zscore'] = \
            [r2_zscore, 0.0]
        metrics_collated['r2_zscore_mae'] = \
            [np.abs(r2_zscore-1), 0.0]
    metrics_collated['r2'] = \
        [r2, 0.0]
    metrics_collated['r2_mae'] = \
        [np.abs(r2-1), 0.0]

    return metrics_collated, metrics_raw, graph_li, curve_pred_li, curve_true_li

from dtaidistance import dtw
def get_metrics(curve_pred, curve_true, is_monotonic):
    '''
    Args:
        curve_pred: (batch_size, 2, resolution) curve
        curve_true: (batch_size, 2, resolution) curve
    Returns:
        dictionary of metric values
        {
            metric: (batch_size,) value,
            ...
        }
    '''
    # norm_log = lambda x: np.log10(np.clip(x, a_min=1e-16, a_max=None))
    pt_li, p_magnitude_li, t_magnitude_li, pt_shape_li = [], [], [], []
    for pred, true in zip(to_numpy(curve_pred), to_numpy(curve_true)):
        p, t = pred[:, 0], true[:, 0]
        p_magnitude, t_magnitude = p.max() - p.min(), t.max() - t.min()
        p_shape = (p - p.min()) / (p_magnitude+1e-12)
        t_shape = (t - t.min()) / (t_magnitude+1e-12)
        pt_li.append((p,t))
        p_magnitude_li.append(p_magnitude)
        t_magnitude_li.append(t_magnitude)
        pt_shape_li.append((p_shape,t_shape))

    metrics = {
        'mse_all':
            [
                get_mse(p, t) \
                for p, t in pt_li
            ],
        'mse_shape':
            [
                get_mse(p, t) \
                for p, t in pt_shape_li
            ],
        'mse_magnitude':
            [
                get_mse(p, t) \
                for p, t in zip(p_magnitude_li, t_magnitude_li)
            ],
        'mae_all':
            [
                get_mae(p, t) \
                for p, t in pt_li
            ],
        'mae_shape':
            [
                get_mae(p, t) \
                for p, t in pt_shape_li
            ],
        'mae_magnitude':
            [
                get_mae(p, t) \
                for p, t in zip(p_magnitude_li, t_magnitude_li)
            ],
        'jaccard_all':
            [
                get_jaccard_index_monotonic(p,t)
                for p, t in pt_li
            ],
        'jaccard_shape':
            [
                get_jaccard_index_monotonic(p,t)
                for p, t in pt_shape_li
            ],
        'mape_magnitude':
            [
                get_mape(p, t) \
                for p, t in zip(p_magnitude_li, t_magnitude_li)
            ],
        # 'r^2_magnitude':
        #     r_squared,
        # 'dtwdist_shape':
        #     mean([
        #         dtw.distance_fast(
        #             to_numpy(get_shape(p)).astype(np.double)[:,0],
        #             to_numpy(get_shape(t)).astype(np.double)[:,0],
        #             use_pruning=True
        #         ) for p, t in zip(curve_pred, curve_true)
        #     ]),
        # 'mape_y_shape':
        #     mean([
        #         get_mape(to_numpy(get_shape(p)), to_numpy(get_shape(t))) \
        #         for p, t in zip(curve_pred, curve_true)
        #     ]),
    }
    metrics['neg_jaccard_all'] = [-x for x in metrics['jaccard_all']]
    return metrics

def collate_metrics(metrics):
    metrics_collated = {}
    for k,v in metrics.items():
        metrics_collated[k] = (np.mean(np.array([x for x in v])), np.std(np.array([x for x in v])))
    return metrics_collated
