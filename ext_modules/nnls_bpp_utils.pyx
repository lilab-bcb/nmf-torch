# cython: language_level=3

import numpy as np
import torch

cimport cython

ctypedef unsigned char uint8

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _nnls_bpp(float[:, :] CTC, float[:, :] CTB, float[:, :] X, device_type):
    cdef int q = CTB.shape[0]
    cdef int r = CTB.shape[1]
    cdef int max_iter = 5 * q

    X[:] = 0.0
    cdef float[:, :] Y = np.zeros_like(CTB) - CTB

    cdef int backup_cap = 3
    cdef int[:] alpha = np.full(r, backup_cap, dtype=np.int32)
    cdef int[:] beta = np.full(r, q+1, dtype=np.int32)

    cdef uint8[:, :] F = np.zeros((q, r), dtype=np.bool_)
    cdef uint8[:, :] V = Y < np.zeros_like(Y, dtype=np.float32)
    cdef int[:] Vsize = np.sum(V, axis=0, dtype=np.int32)

    cdef int i, j
    I = []
    for i in range(Vsize.size):
        if Vsize[i] > 0:
            I.append(i)

    cdef int n_iter, nF, nG, sum_v
    n_iter = 0
    while len(I) > 0 and n_iter < max_iter:
        # Split indices in I into 3 cases:
        FI = []
        for i in range(len(I)):
            if Vsize[i] < beta[i]:
                # Case 1: Apply full exchange rule
                alpha[i] = backup_cap
                beta[i] = Vsize[i]
                F[:, i] ^= V[:, i]
            elif alpha[i] > 0:
                # Case 2: Retry with full exchange rule
                alpha[i] -= 1
                F[:, i] ^= V[:, i]
            else:
                # Case 3: Apply backup rule
                coords = np.nonzero(V[:, i])[0]
                row_idx = coords[coords.size - 1]
                F[row_idx, i] ^= True

            FI.append(F[:, i])

        # solve grouped normal equations
        uniq_F, indices = _get_unique_columns(FI)
        for i in range(uniq_F.shape[1]):
            Fvec = uniq_F[:, i]
            Ii = I[indices == i]
            nF = np.sum(Fvec)
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

        for i in range(I.size):
            v_i = ((X[:, i] < 0.0) & F[:, i]) | ((Y[:, i] < 0.0) & (~F[:, i]))
            vsize_i = np.sum(v_i, dtype=np.int32)
            V[:, i] = v_i
            Vsize[i] = vsize_i
        I = I[Vsize_I > 0]
        V_I = ((X[:, I] < np.zeros((X.shape[0], I.size))) & F[:, I]) | ((Y[:, I] < np.zeros((Y.shape[0], I.size))) & (~F[:, I]))
        Vsize_I = np.sum(V_I, axis=0, dtype=np.int32)
        V[:, I] = V_I
        Vsize[I] = Vsize_I
        I = I[Vsize_I > 0]

        n_iter += 1

    return n_iter if I.size == 0 else -1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _get_unique_columns(int[:, :] mat):
    cdef int[:] indices = np.full(mat.shape[1], -1, dtype=np.int32)
    unique_cols = []

    cdef int i
    cdef int idx = 0
    for j in range(mat.shape[1]):
        if len(unique_cols) == 0:
            unique_cols.append(mat[:, j])
            indices[j] = idx
            idx += 1
        else:
            for i in range(len(unique_cols)):
                if unique_cols[i] == mat[:, j]:
                    indices[j] = i
                    break
            if indices[j] == -1:
                indices[j] = idx
                idx += 1

    return unique_cols, indices
