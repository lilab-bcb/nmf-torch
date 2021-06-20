### NNLS Block principal pivoting: Kim and Park et al 2011.

import numpy as np
import torch

from nmf.cylib.nnls_bpp_utils import _nnls_bpp

def nnls_bpp(CTC, CTB, X, device_type) -> int:
    """
        min ||CX-B||_F^2
        KKT conditions:
           1) Y = CTC @ X - CTB
           2) Y >= 0, X >= 0
           3) XY = 0
        Return niter; if niter < 0, nnls_bpp does not converge.
    """
    # CTC = C.T @ C
    # CTB = C.T @ B

    # dtype = C.dtype
    # X = torch.zeros((q, r), dtype=dtype)

    if device_type == 'cpu':
        return _nnls_bpp(CTC.numpy(), CTB.numpy(), X.numpy(), 'cpu')
    else:
        X_cpu = torch.zeros_like(X, device='cpu')
        n_iter = _nnls_bpp(CTC.cpu().numpy(), CTB.cpu().numpy(), X_cpu.numpy(), 'gpu')
        X[:] = X_cpu.cuda()
        return n_iter
