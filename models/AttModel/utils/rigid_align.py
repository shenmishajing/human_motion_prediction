#!/usr/bin/env python
# encoding: utf-8
import torch

# from IPython import embed


def rigid_transform_3D(A, B):
    centroid_A = A.mean(dim=-2, keepdim=True)
    centroid_B = B.mean(dim=-2, keepdim=True)
    H = (A - centroid_A).mT.matmul(B - centroid_B)
    U, s, V = torch.linalg.svd(H)
    R = U.matmul(V).mT
    V[..., 2, :] = -V[..., 2, :]
    NR = U.matmul(V).mT
    R = R.where((torch.linalg.det(R) > 0)[..., None, None], NR)
    t = centroid_B - centroid_A.matmul(R.mT)
    return R, t


def _rigid_align(A, B):
    R, t = rigid_transform_3D(A, B)
    return A.matmul(R.mT) + t


def rigid_align(A, B):
    # align A to B
    # A,B:torch.Size([ba, nb_frames, 36, 3])
    A = A.reshape(*A.shape[:2], -1, 18, 3)
    B = B.reshape(*B.shape[:2], -1, 18, 3)
    res = [_rigid_align(A[:, :, i], B[:, :, i])[:, :, None] for i in range(A.shape[2])]
    return torch.cat(res, dim=2).reshape(*A.shape[:2], -1, 3)
