import torch

from src.config import ETH_FULL_C_VECTOR

DEFAULT_MAGNITUDE = 0.0

def get_curve(c, curve_method='max'):
    if curve_method == 'max':
        return get_curve_max(c)
    elif curve_method == 'unorm_last':
        return get_curve_unorm_last(c)
    elif curve_method == 'unorm_max':
        return get_curve_unorm_max(c)
    elif curve_method == 'slope':
        return get_curve_slope(c)
    elif curve_method == 'unorm_slope':
        return get_curve_unorm_slope(c)
    elif curve_method is None:
        return DEFAULT_MAGNITUDE * torch.ones(1, dtype=torch.float), torch.FloatTensor(c[:,1:])
    else:
        assert False

def get_curve_unorm_max(c):
    # c = torch.tensor(upsample(c, resolution))
    c = torch.FloatTensor(c)
    c_magnitude = torch.max(torch.abs(c[:,1:]), dim=0)[0]
    c_shape = c[:,1:]/(c_magnitude.reshape(1,-1) + 1e-12)
    return c_magnitude, c_shape

def get_curve_unorm_last(c):
    # c = torch.tensor(upsample(c, resolution))
    c = torch.FloatTensor(c)
    c_magnitude = c[-1,1:]
    c_shape = c[:,1:]/(c_magnitude.reshape(1,-1) + 1e-12)
    return c_magnitude, c_shape

def get_curve_max(c):
    # c = torch.tensor(upsample(c, resolution))
    c = torch.tensor(c)
    c = torch.nan_to_num(c, 0.0)
    c_max = torch.max(torch.abs(c[:,1:]), dim=0)[0]
    if ETH_FULL_C_VECTOR:
        c_magnitude = torch.log10(c_max+1.0)
    else:
        c_magnitude = torch.log10(c_max)
    c_shape = c[:,1:]/(c_max.reshape(1,-1)+1e-12)
    return c_magnitude, c_shape

def get_curve_unorm_slope(c):
    c = torch.FloatTensor(c)
    n_steps = 8
    c_magnitude = torch.mean(c[1:n_steps+1, 1:] - c[:n_steps, 1:], dim=0)
    assert all(torch.gt(c_magnitude, 0.0))
    c_shape = c[:,1:]/c_magnitude.reshape(1,-1)
    return c_magnitude, c_shape

def get_curve_slope(c):
    # c = torch.tensor(upsample(c, resolution))
    c = torch.FloatTensor(c)
    n_steps = 8
    c_max = torch.mean(c[1:n_steps+1, 1:] - c[:n_steps, 1:], dim=0)
    # c_max = torch.max(torch.abs(c[:,1:]), dim=0)[0]
    assert all(torch.gt(c_max, 0.0))
    c_magnitude = torch.log10(c_max)#+1.0)
    c_shape = c[:,1:]/c_max.reshape(1,-1)
    return c_magnitude, c_shape

def unnormalize_curve(c_magnitude, c_shape, curve_method='max', **kwargs):
    c_max = torch.pow(10, c_magnitude.unsqueeze(1)) #-1.0
    if curve_method is None:
        cmin, cmax = None, None
        c = c_shape#*c_max
    else:
        if 'unorm' in curve_method:
            c_max = c_magnitude.unsqueeze(1) #-1.0
        cmin = kwargs['cmin'] if 'cmin' in kwargs else c_shape.min(dim=1)[0].unsqueeze(1)
        cmax = 1+cmin#kwargs['cmax'] if 'cmax' in kwargs else c_shape.max(dim=1)[0].unsqueeze(1)
        c_shape = (c_shape-cmin)/(cmax-cmin)
        c = c_shape*c_max
    return c, cmin, cmax
