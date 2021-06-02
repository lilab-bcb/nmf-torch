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

    return _nnls_bpp(CTC.numpy(), CTB.numpy(), X.numpy(), device_type)
