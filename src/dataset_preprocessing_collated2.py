import numpy as np
import os
import pickle
from scipy.interpolate import CubicSpline
from copy import deepcopy

def get_ood_data(X, resolution=256):
    # xrange frequencies: 1000 to 12000
    # idx = int(3000 / 11000 * (resolution - 1))
    idx1 = int(0 / 11000 * (resolution - 1))
    idx2 = int(2000 / 11000 * (resolution - 1))
    # idx3 = int(2000 / 11000 * (resolution - 1))
    idx4 = int(4000 / 11000 * (resolution - 1))
    idx_last = resolution - 1

    xx = np.array([
        [idx1],
        [idx2],
        # [idx3],
        [idx4],
        [idx_last]
    ])
    yy = np.array([[1.0], [0.0], [1.0], [1.0]])
    x_range = np.arange(X.shape[0])
    scale = np.array([CubicSpline(xx[:,i], yy[:,i])(x_range) for i in range(xx.shape[1])]).transpose()
    # scale[idx2:idx3] = 0.0
    scale[idx4:] = 1.0
    out = np.abs(scale) * X
    return out

def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    out = np.array([CubicSpline(xx[:,i], yy[:,i])(x_range) for i in range(xx.shape[1])]).transpose()
    return out

def timewarp(x, sigma):
    x_range = np.arange(x.shape[0])
    tt = GenerateRandomCurves(x, sigma) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    out = []
    for i in range(x.shape[1]):
        t_scale = (x.shape[0] - 1) / tt_cum[-1, i]
        tt_cum[:, i] = tt_cum[:, i] * t_scale
        out.append(np.interp(x_range, tt_cum[:, i], x[:, i]))
    return np.stack(out, axis=1)

def main():
    # below_minus10db_1to2k = [130, 139, 170, 195, 433, 435, 470, 594, 595, 610, 667, 719, 816, 869, 1027, 1065, 1102, 1150, 1240, 1323, 1325, 1363, 1532, 1536, 1616, 1632, 1658, 1662, 1708, 1712, 1738, 1823, 1977, 1999, 2022, 2042, 2203, 2208, 2225, 2229, 2258, 2263, 2267, 2291, 2375, 2378, 2460, 2462, 2464, 2471, 2544, 2659, 2685, 2844, 2845, 2870]
    # below_minus10db_1to4k = [130, 139, 195, 435, 470, 594, 667, 816, 1102, 1150, 1323, 1325, 1532, 1536, 1616, 1632, 1658, 1662, 1738, 1977, 1999, 2022, 2042, 2203, 2208, 2225, 2263, 2267, 2291, 2375, 2378, 2464, 2471, 2544, 2844, 2845, 2870]
    # below_minus20db_1to4k = [1150, 1536, 1999, 2022]
    num_aug = 0 #5
    cid_offset = 2955 # TODO: hardcoded
    cid2gid = lambda x: x # TODO: hardcoded
    valid_cid_li = []
    pn = '/home/derek/Documents/MaterialSynthesis/data/marco_02142024_PSU/preprocessed_unitcell_True_mixed'
    dn_old = os.path.join(pn, 'train')
    dn_new = os.path.join(pn, 'train2')
    for fn in os.listdir(os.path.join(dn_old, 'curves')):
        cid = int(fn.split('.')[0])
        with open(os.path.join(os.path.join(dn_old, 'curves'), fn), 'rb') as fp:
            # c = pickle.load(fp)
            # if np.all(c['curve'][:int(2000/11000*256), 1] < 0.0):
            valid_cid_li.append(cid)
    print(valid_cid_li)

    os.system(f'rm -r {dn_new}')
    os.system(f'mkdir {dn_new}')
    os.system(f'mkdir {os.path.join(dn_new, "curves")}')
    os.system(f'mkdir {os.path.join(dn_new, "graphs")}')
    fp_match = open(os.path.join(dn_new, 'mapping.tsv'), 'w+')

    # def get_cid(valid_cid_li, cid_offset):
    #     if len(valid_cid_li) > 0:
    #         cur_cid = valid_cid_li.pop()
    #     else:
    #         cid_offset += 1
    #         cur_cid = cid_offset
    #     return cur_cid, valid_cid_li, cid_offset
    # fp_graph_old = os.path.join(dn_old, 'graphs', f'{valid_cid_li[0]}.gpkl')
    # fp_graph_new = os.path.join(dn_new, 'graphs', f'{valid_cid_li[0]}.gpkl')
    # with open(os.path.join(os.path.join(dn_old, 'curves'), f'{valid_cid_li[0]}.pkl'), 'rb') as fp:
    #     c = pickle.load(fp)
    # template = deepcopy(c)
    # os.system(f'cp {fp_graph_old} {fp_graph_new}')
    # for thresh in [-10, -20, -30, -40]:
    #
    #     cur_cid, valid_cid_li, cid_offset = get_cid(valid_cid_li, cid_offset)
    #     template['curve'][:, 1] *= 0.0
    #     template['curve'][:int(3000/11000*256), 1] += thresh
    #     with open(os.path.join(os.path.join(dn_new, 'curves'), f'{cur_cid}.pkl'), 'wb') as fp:
    #         pickle.dump(template, fp)
    #     fp_match.writelines([f'{valid_cid_li[0]}\t{cur_cid}\n'])
    #
    #     for first_peak in range(4,11):
    #         cur_cid, valid_cid_li, cid_offset = get_cid(valid_cid_li, cid_offset)
    #         template['curve'][int(first_peak*1000/11000*256):int((first_peak+1)*1000/11000*256), 1] += thresh
    #         with open(os.path.join(os.path.join(dn_new, 'curves'), f'{cur_cid}.pkl'), 'wb') as fp:
    #             pickle.dump(template, fp)
    #         fp_match.writelines([f'{valid_cid_li[0]}\t{cur_cid}\n'])
    #
    #         if first_peak < 10:
    #             for second_peak in range(first_peak+2, 11):
    #                 cur_cid, valid_cid_li, cid_offset = get_cid(valid_cid_li, cid_offset)
    #                 template['curve'][int(second_peak*1000/11000*256):int((second_peak+1)*1000/11000*256), 1] += thresh
    #                 with open(os.path.join(os.path.join(dn_new, 'curves'), f'{cur_cid}.pkl'), 'wb') as fp:
    #                     pickle.dump(template, fp)
    #                 fp_match.writelines([f'{valid_cid_li[0]}\t{cur_cid}\n'])
    #                 template['curve'][int(second_peak*1000/11000*256):int((second_peak+1)*1000/11000*256), 1] -= thresh
    #
    #         template['curve'][int(first_peak*1000/11000*256):int((first_peak+1)*1000/11000*256), 1] -= thresh

    for cid in valid_cid_li:
        with open(os.path.join(os.path.join(dn_old, 'curves'), f'{cid}.pkl'), 'rb') as fp:
            c = pickle.load(fp)
        assert c['curve'].shape[1] == 2

        c['curve'][:, 1:] = get_ood_data(c['curve'][:, 1:] + 40.0) - 40.0
        with open(os.path.join(os.path.join(dn_new, 'curves'), f'{cid}.pkl'), 'wb') as fp:
            pickle.dump(c, fp)

        gid = cid2gid(cid)
        fp_graph_old = os.path.join(dn_old, 'graphs', f'{gid}.gpkl')
        fp_graph_new = os.path.join(dn_new, 'graphs', f'{gid}.gpkl')
        os.system(f'cp {fp_graph_old} {fp_graph_new}')
        fp_match.writelines([f'{gid}\t{cid}\n'])

        for _ in range(num_aug):
            cid_offset += 1
            tmp = deepcopy(c['curve'][:, 1:])
            c['curve'][:, 1:] += 40.0
            c['curve'][:, 1:] = get_ood_data(c['curve'][:, 1:])
            Z = np.expand_dims(c['curve'][:, 1:].max(axis=0),0)
            assert max(Z) > 0.0
            c['curve'][:, 1:] /= Z
            c['curve'][:, 1:] = timewarp(c['curve'][:, 1:], 0.2)
            c['curve'][:, 1:] *= Z
            c['curve'][:, 1:] -= 40.0
            with open(os.path.join(os.path.join(dn_new, 'curves'), f'{cid_offset}.pkl'), 'wb') as fp:
                pickle.dump(c, fp)
            c['curve'][:, 1:] = tmp
            fp_match.writelines([f'{cid}\t{cid_offset}\n'])
    fp_match.close()


if __name__ == '__main__':
    main()

    import matplotlib.pyplot as plt
    dn = '/home/derek/Documents/MaterialSynthesis/data/marco_02142024_PSU/preprocessed_unitcell_True_mixed/test/curves'
    for fn in os.listdir(dn):
        with open(os.path.join(dn, fn), 'rb') as fp:
            c = pickle.load(fp)
            plt.plot(c['curve'][:,0], np.clip(c['curve'][:,1], a_min=-40.0, a_max=9999999.99), alpha=0.7)
    plt.savefig('tmp_normal.png')
    plt.clf()

    dn = '/home/derek/Documents/MaterialSynthesis/data/marco_02142024_PSU/preprocessed_unitcell_True_mixed/test2/curves'
    for fn in os.listdir(dn):
        with open(os.path.join(dn, fn), 'rb') as fp:
            c = pickle.load(fp)
            plt.plot(c['curve'][:,0], np.clip(c['curve'][:,1], a_min=-40.0, a_max=9999999.99), alpha=0.7)
    plt.savefig('tmp_ood.png')
