"""
adapted from : https://github.com/wei-mao-2019/HisRepItself/blob/master/model/GCN.py
under MIT license.
"""
# util.py

import math

import torch


def get_dct_matrix(N, tensor=None):
    if tensor is None:
        dtype = None
        device = None
    else:
        dtype = tensor.dtype
        device = tensor.device
    i = torch.arange(N, dtype=dtype, device=device)[None, :]
    j = torch.arange(N, dtype=dtype, device=device)[:, None]
    dct_m = math.sqrt(1 / N) * torch.cos(math.pi * (i + 1 / 2) * j / N)
    mask = torch.ones_like(j)
    mask[0] = 0
    dct_m *= (math.sqrt(2) - 1) * mask + 1
    idct_m = torch.linalg.inv(dct_m)
    return dct_m, idct_m
