import torch.nn as nn
import numpy as np
import torch
import math

from src.config import ETH_FULL_C_VECTOR, args
from src.utils import MLP
from src.layers_pooler import Merger


class MLPWrapper(nn.Module):
    def __init__(self, dim_in, dim_out, n_layers=5, **kwargs):
        super().__init__()
        if dim_in >= dim_out:
            multipliers = np.zeros(n_layers+1)
            dim_li = np.clip(dim_in / (2**multipliers), a_min=dim_out, a_max=None)
            dim_li[-1] = dim_out
            dim_li = dim_li.astype(int)
        else:
            dim_li = [dim_out for _ in range(n_layers+1)]
            dim_li[0] = dim_in
        self.mlp = MLP(dim_li, **kwargs)
        self.bn = nn.LayerNorm(dim_in)
        self.clamp = DifferentiableClamp().apply

    def forward(self, x):
        x = self.mlp(x)
        return x

class MLPDecoder(nn.Module):
    def __init__(self, dim_in, dim_out, c_stats=None, merge_name=None, n_layers=5, **kwargs):
        super().__init__()
        self.merger = Merger(dim_in, dim_in, merge_name, num_inputs=2)
        self.c_stats = c_stats
        if dim_in >= dim_out:
            multipliers = np.zeros(n_layers+1) # np.arange(n_layers)
            dim_li = np.clip(dim_in / (2**multipliers), a_min=dim_out, a_max=None)
            dim_li[-1] = dim_out
            dim_li = dim_li.astype(int)
        else:
            dim_li = [dim_out for _ in range(n_layers+1)]
            dim_li[0] = dim_in
        if ETH_FULL_C_VECTOR:
            dim_li[-1] = 10
            self.mlp = MLP(dim_li, **kwargs)
        else:
            assert dim_li[-1] == 2
            dim_li[-1] = 1
            self.mlp_C = MLP(dim_li, **kwargs)
            self.mlp_n = MLP(dim_li, **kwargs)
        self.bn = nn.BatchNorm1d(dim_in)
        self.clamp = DifferentiableClamp().apply

    def forward(self, emb_nodes, emb_edges, rho, return_cn=False, **kwargs):
        # magn = C*rho^n
        # magn * (256/0.3) = C*(256/0.3)*rho^n
        # log magn = log(C) + log(256/0.3) + n*log(rho)
        # log magn - mean = (log(C) - mean + log(256/0.3)) + n*log(rho)
        # (log magn - mean)/std = (log(C) - mean + log(256/0.3))/std + n/std*log(rho)
        x = self.merger(emb_nodes, emb_edges)
        if ETH_FULL_C_VECTOR:
            x = self.mlp(x)
        else:
            cn = torch.cat((self.mlp_C(x), self.mlp_n(x)), dim=-1)

            c_magnitude_mean, c_magnitude_std, *_ = self.c_stats
            c_magnitude_mean, c_magnitude_std = c_magnitude_mean.to(x.device), c_magnitude_std.to(x.device)
            # C_min = (math.log10(1e-12) - c_magnitude_mean + math.log10(256/0.3))/c_magnitude_std
            C_min = (math.log10(1e-10) - c_magnitude_mean)/c_magnitude_std
            # C_max = (math.log10(2.0)-c_magnitude_mean + math.log10(256/0.3))/c_magnitude_std
            n_min = 1.0/c_magnitude_std
            # n_max = 3.0/c_magnitude_std

            if args['dataset']['curve_norm_cfg']['curve_method'] in ['max','slope']:
                cn_ = torch.clamp(cn, min=torch.cat((C_min, n_min)))#, max=torch.cat((C_max, n_max)))
                cn = cn + (cn_ - cn).detach()
            x = \
                cn * torch.stack((
                    torch.ones_like(rho, dtype=torch.float32, device=x.device),
                    torch.log10(torch.clamp(rho.to(device=x.device), min=0.02))), dim=-1)
            x = torch.sum(x, dim=-1).unsqueeze(-1) + np.log10(4/7)

            if return_cn:
                cn = cn.detach()
                cn[:,0] = torch.pow(10, c_magnitude_std*(cn[:,0]) + c_magnitude_mean)# - math.log10(256/0.3))
                cn[:,1] = c_magnitude_std*cn[:,1]
                return x, cn
        return x

class DifferentiableClamp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0.0, max=2.0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()