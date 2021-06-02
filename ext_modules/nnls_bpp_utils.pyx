# cython: language_level=3

### NNLS Block principal pivoting: Kim and Park et al 2011.

import numpy as np
import torch

cimport cython

ctypedef unsigned char uint8


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _nnls_bpp(float[:, :] CTC, float[:, :] CTB, float[:, :] X, str device_type):
    # CTC = C.T @ C
    # CTB = C.T @ B

    # dtype = C.dtype
    # X = torch.zeros((q, r), dtype=dtype)

    cdef Py_ssize_t i, j, k, l, m, uniq_idx
    cdef int q = CTB.shape[0]
    cdef int r = CTB.shape[1]
    cdef int max_iter = 5 * q

    ### Initialization, setting G = 1-q
    for i in range(q):
        for j in range(r):
            X[i, j] = 0.0

    cdef float[:, :] Y = np.zeros_like(CTB, dtype=np.float32)
    for i in range(q):
        for j in range(r):
            Y[i, j] = -CTB[i, j]

    cdef int backup_cap = 3
    cdef int[:] alpha = np.full(r, backup_cap, dtype=np.int32)  # cap on back up rule
    cdef int[:] beta = np.full(r, q+1, dtype=np.int32)  # number of infeasible variables

    cdef uint8[:, :] F = np.zeros((q, r), dtype=np.bool_)  # y_F = 0, G = ~F, x_G = 0

    cdef uint8[:, :] V = np.zeros((q, r), dtype=np.bool_)
    for i in range(q):
        for j in range(r):
            V[i, j] = Y[i, j] < 0

    cdef int[:] Vsize = np.zeros((r,), dtype=np.int32)
    for i in range(q):
        for j in range(r):
            Vsize[j] += V[i, j]

    cdef int[:] I = np.zeros((r,), dtype=np.int32)  # infeasible columns
    cdef int size_I = 0
    for i in range(r):
        if Vsize[i] > 0:
            I[size_I] = i
            size_I += 1

    cdef uint8[:, :] uniq_F = np.zeros((q, r), dtype=np.bool_)
    cdef int[:] indices = np.zeros((r,), dtype=np.int32)
    cdef uint8[:] Fvec = np.zeros((q,), dtype=np.bool_)
    cdef int[:] Ii = np.zeros((r,), dtype=np.int32)
    cdef float[:, :] CTC_L = np.zeros((q, q), dtype=np.float32)
    cdef float[:, :] CTB_L = np.zeros((q, r), dtype=np.float32)
    cdef uint8[:, :] V_I = np.zeros((q, r), dtype=np.bool_)
    cdef int[:] Vsize_I = np.zeros((r,), dtype=np.int32)
    cdef float[:, :] y = np.zeros((q, r), dtype=np.float32)

    cdef int col_idx, row_idx, nF, nG, size_uniq_F, uniq_flag, size_Ii, CTC_L_M, CTC_L_N, CTB_L_M, CTB_L_N
    cdef int n_iter = 0
    while size_I > 0 and n_iter < max_iter:
        # Split indices in I into 3 cases:
        for j in range(size_I):
            col_idx = I[j]
            if Vsize[col_idx] < beta[col_idx]:
                # Case 1: Apply full exchange rule
                alpha[col_idx] = backup_cap
                beta[col_idx] = Vsize[col_idx]
                for i in range(q):
                    F[i, col_idx] ^= V[i, col_idx]
            elif alpha[col_idx] > 0:
                # Case 2: Retry with full exchange rule
                alpha[col_idx] -= 1
                for i in range(q):
                    F[i, col_idx] ^= V[i, col_idx]
            else:
                # Case 3: Apply backup rule
                row_idx = 0
                for i in range(q-1, -1, -1):
                    if V[i, col_idx] > 0:
                        row_idx = i
                        break
                F[row_idx, col_idx] ^= True

        # Get unique F columns with indices mapping back to F.
        size_uniq_F = 0
        for j in range(size_I):
            uniq_flag = 1
            for i in range(size_uniq_F):
                if _equal_column(uniq_F, i, F, I[j]):
                    # Not unique, only record index.
                    indices[j] = i
                    uniq_flag = 0
                    break

            if uniq_flag:
                # Unique column.
                for i in range(uniq_F.shape[0]):
                    uniq_F[i, size_uniq_F] = F[i, I[j]]
                indices[j] = size_uniq_F
                size_uniq_F += 1

        # Solve grouped normal equations
        for uniq_idx in range(size_uniq_F):
            for i in range(uniq_F.shape[0]):
                Fvec[i] = uniq_F[i, uniq_idx]

            # Ii = I[indices == uniq_idx]
            size_Ii = 0
            for i in range(size_I):
                if indices[i] == uniq_idx:
                    Ii[size_Ii] = I[i]
                    size_Ii += 1

            nF = 0
            for i in range(Fvec.size):
                nF += Fvec[i]
            nG = q - nF

            if nF > 0:
                CTC_L_M = 0
                for i in range(q):
                    CTC_L_N = 0
                    if Fvec[i]:
                        for j in range(q):
                            if Fvec[j]:
                                CTC_L[CTC_L_M, CTC_L_N] = CTC[i, j]
                                CTC_L_N += 1
                        CTC_L_M += 1
                assert CTC_L_M == CTC_L_N, "CTC submatrix is not square!"

                L = torch.cholesky(torch.tensor(CTC_L[0:CTC_L_M, 0:CTC_L_N], device=device_type))

                CTB_L_M = 0
                for i in range(q):
                    CTB_L_N = 0
                    if Fvec[i]:
                        for j in range(size_Ii):
                            CTB_L[CTB_L_M, CTB_L_N] = CTB[i, Ii[j]]
                            CTB_L_N += 1
                        CTB_L_M += 1

                x = torch.cholesky_solve(torch.tensor(CTB_L[0:CTB_L_M, 0:CTB_L_N], device=device_type), L)
                x = x.numpy() if device_type == 'cpu' else x.cpu().numpy()

                k = 0
                for i in range(q):
                    if Fvec[i]:
                        for j in range(size_Ii):
                            X[i, Ii[j]] = x[k, j] if np.abs(x[k, j]) >= 1e-12 else 0.0
                            Y[i, Ii[j]] = 0.0
                        k += 1

            if nG > 0:
                k = 0
                for i in range(q):
                    if not Fvec[i]:
                        for j in range(x.shape[1]):
                            y[k, j] = 0.0
                            if nF > 0:
                                # CTC[~Fvec, Fvec] @ x
                                m = 0
                                for l in range(q):
                                    if Fvec[l]:
                                        y[k, j] += CTC[k, l] * x[m, j]
                                        m += 1
                            y[k, j] -= CTB[i, Ii[j]]
                            if np.abs(y[k, j]) < 1e-12:
                                y[k, j] = 0.0
                            Y[i, Ii[j]] = y[k, j]
                            X[i, Ii[j]] = 0.0
                        k += 1

        for j in range(size_I):
            Vsize_I[j] = 0
            for i in range(V_I.shape[0]):
                V_I[i, j] = ((X[i, I[j]] < 0.0) & F[i, I[j]]) | ((Y[i, I[j]] < 0.0) & (~F[i, I[j]]))
                Vsize_I[j] += V_I[i, j]
                V[i, I[j]] = V_I[i, j]
            Vsize[I[j]] = Vsize_I[j]

        old_size_I = size_I
        size_I = 0
        for i in range(old_size_I):
            if Vsize_I[i] > 0:
                I[size_I] = I[i]
                size_I += 1

        n_iter += 1

    return n_iter if I.size == 0 else -1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _equal_column(uint8[:, :] mat_A, int col_A, uint8[:, :] mat_B, int col_B):
    for i in range(mat_A.shape[0]):
        if mat_A[i, col_A] != mat_B[i, col_B]:
            return 0

    return 1
