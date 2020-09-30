import numpy as np
from scipy.linalg import hadamard
import torch

def _is_power_2(n):
    return int(2**int(np.log(n) / np.log(2)) == n)


def _srht(indices, v):
    n = v.shape[0]
    if n == 1:
        return v
    i1 = indices[indices < n//2]
    i2 = indices[indices >= n//2]
    if len(i1) == 0:
        return _srht(i2-n//2, v[:n//2,::]-v[n//2:,::])
    elif len(i2) == 0:
        return _srht(i1, v[:n//2,::]+v[n//2:,::])
    else:
        return torch.cat([_srht(i1, v[:n//2,::]+v[n//2:,::]), _srht(i2-n//2, v[:n//2,::]-v[n//2:,::])], dim=0)


def _compute_sketch_matrix(indices, signs):
    n = signs.shape[0]
    k = int(np.log(n) / np.log(2)) + 1
    H = torch.Tensor([[1.]])
    ii = 1
    while ii < k:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H,-H], dim=1)], dim=0) / np.sqrt(2)
        ii += 1
    H = H * signs
    return H[indices]



def srht(matrix, sketch_size):
    if matrix.ndim == 1:
        matrix = matrix.reshape((-1,1))
    n = matrix.shape[0]
    if not _is_power_2(n):
        new_dim = 2**(int(np.log(n) / np.log(2))+1)
        matrix = torch.cat([matrix, torch.zeros((new_dim - n, matrix.shape[1]))], dim=0)
    indices = torch.Tensor(np.sort(np.random.choice(np.arange(matrix.shape[0]), sketch_size, replace=False))).long()
    signs = torch.Tensor(np.random.choice([-1,1], matrix.shape[0], replace=True).reshape((-1,1))).long()
    return 1./np.sqrt(sketch_size)*_srht(indices, signs*matrix), np.sqrt(n/sketch_size)*_compute_sketch_matrix(indices, signs.reshape((-1,)))[::,:n]



