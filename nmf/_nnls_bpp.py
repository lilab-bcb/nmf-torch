### NNLS Block principal pivoting: Kim and Park et al 2011.

import numpy as np
import torch

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

    q, r = CTB.shape
    max_iter = q * 5

    ### Initialization, setting G = 1-q
    X[:] = 0.0
    Y = -CTB

    backup_cap = 3
    alpha = torch.full([r], backup_cap, dtype=torch.int, device=device_type) # cap on back up rule
    beta = torch.full([r], q+1, dtype=torch.int, device=device_type) # number of infeasible variables

    F = torch.zeros((q, r), dtype=bool, device=device_type) # y_F = 0, G = ~F, x_G = 0

    V = Y < 0.0
    Vsize = V.sum(axis = 0, dtype = torch.int, device=device_type)
    I = torch.where(Vsize > 0)[0] # infeasible columns

    n_iter = 0
    while I.shape[0] > 0 and n_iter < max_iter:
        # Case 1: Apply full exchange rule
        idx = Vsize[I] < beta[I]
        col_idx = I[torch.where(idx)[0]]
        if col_idx.shape[0] > 0:
            alpha[col_idx] = backup_cap
            beta[col_idx] = Vsize[col_idx]
            F[:, col_idx] ^= V[:, col_idx]
        neg_idx = ~idx

        # Case 2: Retry with full exchange rule
        idx2 = alpha[I] > 0
        col_idx = I[torch.where(neg_idx & idx2)]
        if col_idx.shape[0] > 0:
            alpha[col_idx] -= 1
            F[:, col_idx] ^= V[:, col_idx]

        # Case 3: Apply backup rule
        col_idx = I[torch.where(neg_idx & (~idx2))]
        if col_idx.shape[0] > 0:
            coords = torch.nonzero(V[:, col_idx].T)
            row_idx = torch.tensor([coords[i, 1] for i in range(coords.shape[0]) if i+1==coords.shape[0] or coords[i, 0] < coords[i+1, 0]])
            F[row_idx, col_idx] ^= True

        # solve grouped normal equations
        uniq_F, indices = torch.unique(F[:, I], dim = 1, return_inverse = True)
        for i in range(uniq_F.shape[1]):
            Fvec = uniq_F[:, i]
            Ii = I[indices == i]
            nF = Fvec.sum()
            nG = q - nF
            if nF > 0:
                L = torch.cholesky(CTC[np.ix_(Fvec, Fvec)])
                mesh_idx = np.ix_(Fvec, Ii)
                x = torch.cholesky_solve(CTB[mesh_idx], L)
                x[torch.abs(x) < 1e-12] = 0.0
                X[mesh_idx] = x
                Y[mesh_idx] = 0.0
            if nG > 0:
                mesh_idx = np.ix_(~Fvec, Ii)
                if nF > 0:
                    y = CTC[np.ix_(~Fvec, Fvec)] @ x - CTB[mesh_idx]
                else:
                    y = - CTB[mesh_idx]
                y[torch.abs(y) < 1e-12] = 0.0
                Y[mesh_idx] = y
                X[mesh_idx] = 0.0

        V_I = ((X[:, I] < 0.0) & F[:, I]) | ((Y[:, I] < 0.0) & (~F[:, I]))
        Vsize_I = V_I.sum(axis = 0, dtype = torch.int)
        V[:, I] = V_I
        Vsize[I] = Vsize_I
        I = I[Vsize_I > 0]

        n_iter += 1

    return n_iter if I.shape[0] == 0 else -1
