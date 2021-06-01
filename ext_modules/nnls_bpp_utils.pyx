# cython: language_level=3

import numpy as np
import torch

cimport cython

ctypedef unsigned char uint8

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _nnls_bpp(float[:, :] CTC, float[:, :] CTB, float[:, :] X, device_type):
    cdef Py_ssize_t i, j, k, l, m
    cdef int q = CTB.shape[0]
    cdef int r = CTB.shape[1]
    cdef int max_iter = 5 * q

    X[:] = 0.0
    cdef float[:, :] Y = np.zeros_like(CTB, dtype=np.float32)
    for i in range(q):
        for j in range(r):
            Y[i, j] = -CTB[i, j]

    cdef int backup_cap = 3
    cdef int[:] alpha = np.full(r, backup_cap, dtype=np.int32)
    cdef int[:] beta = np.full(r, q+1, dtype=np.int32)

    cdef uint8[:, :] F = np.zeros((q, r), dtype=np.bool_)

    cdef uint8[:, :] V = np.zeros((q, r), dtype=np.bool_)
    for i in range(q):
        for j in range(r):
            V[i, j] = Y[i, j] < 0

    cdef int[:] Vsize = np.zeros((r,), dtype=np.int32)
    for i in range(q):
        for j in range(r):
            Vsize[j] += V[i, j]

    cdef int[:] I = np.full(r, -1, dtype=np.int32)
    cdef int[:] Ii = np.zeros((r,), dtype=np.int32)
    cdef int size_I = 0
    for i in range(r):
        if Vsize[i] > 0:
            I[size_I] = i
            size_I += 1

    cdef int n_iter, nF, nG
    cdef int[:] coords = np.full(r, -1, dtype=np.int32)
    cdef uint8[:, :] uniq_F = np.zeros((q, r), dtype=np.bool_)
    cdef uint8[:] Fvec = np.zeros((q,), dtype=np.bool_)
    cdef float[:, :] CTC_L = np.zeros((q, q), dtype=np.float32)
    cdef float[:, :] CTB_L = np.zeros((q, r), dtype=np.float32)
    cdef uint8[:, :] V_I = np.zeros((q, r), dtype=np.bool_)
    cdef int[:] Vsize_I = np.zeros((r,), dtype=np.int32)
    cdef int[:] indices = np.zeros((r,), dtype=np.int32)
    cdef float[:, :] y = np.zeros((q, r), dtype=np.float32)
    cdef int size_coords, row_idx, size_uniq_F, uniq_flag, size_Ii, CTC_L_M, CTC_L_N, CTB_L_M, CTB_L_N
    n_iter = 0
    while size_I > 0 and n_iter < max_iter:
        size_coords = 0
        # Split indices in I into 3 cases:
        for j in range(size_I):
            if Vsize[j] < beta[j]:
                # Case 1: Apply full exchange rule
                alpha[j] = backup_cap
                beta[j] = Vsize[j]
                for i in range(q):
                    F[i, j] ^= V[i, j]
            elif alpha[j] > 0:
                # Case 2: Retry with full exchange rule
                alpha[j] -= 1
                for i in range(q):
                    F[i, j] ^= V[i, j]
            else:
                # Case 3: Apply backup rule
                for i in range(q):
                    if V[i, j] > 0:
                        coords[size_coords] = i
                        size_coords += 1
                row_idx = coords[size_coords - 1]
                F[row_idx, j] ^= True

        # solve grouped normal equations
        size_uniq_F = 0
        for j in range(size_I):
            uniq_flag = 1
            for i in range(size_uniq_F):
                if _equal_column(uniq_F, i, F, I[j], uniq_F.shape[0]) == 1:
                    # Not unique, only record index.
                    indices[j] = i
                    uniq_flag = 0
                    break

            if uniq_flag == 1:
                # Unique column.
                for i in range(uniq_F.shape[0]):
                    uniq_F[i, size_uniq_F] = F[i, I[j]]
                indices[j] = size_uniq_F
                size_uniq_F += 1

        for j in range(size_uniq_F):
            for i in range(uniq_F.shape[0]):
                Fvec[i] = uniq_F[i, j]

            size_Ii = 0
            for k in range(size_I):
                if indices[k] == j:
                    Ii[size_Ii] = I[k]
                    size_Ii += 1

            nF = 0
            for i in range(Fvec.shape[0]):
                nF += Fvec[i]
            nG = q - nF

            if nF > 0:
                CTC_L_M = 0
                for i in range(q):
                    CTC_L_N = 0
                    for j in range(q):
                        if Fvec[i] and Fvec[j]:
                            CTC_L[CTC_L_M, CTC_L_N] = CTC[i, j]
                            CTC_L_N += 1
                    CTC_L_M += 1
                assert CTC_L_M == CTC_L_N, "CTC submatrix is not square!"
                L = torch.cholesky(torch.tensor(torch.tensor(CTC_L)))  # TODO: How to feed a submatrix?

                CTB_L_M = 0
                for i in range(q):
                    CTB_L_N = 0
                    if Fvec[i]:
                        for j in range(size_Ii):
                            CTB_L[CTC_L_M, CTC_L_N] = CTB[i, Ii[j]]
                            CTC_L_N += 1
                    CTC_L_M += 1
                x = torch.cholesky_solve(torch.tensor(CTB_L), L).numpy()  # TODO: How to feed a submatrix?
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        if np.abs(x[i, j]) < 1e-12:
                            x[i, j] = 0.0

                k = 0
                for i in range(q):
                    if Fvec[i]:
                        for j in range(size_Ii):
                            X[i, Ii[j]] = x[k, j]
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
                                        y[k, j] += CTC[i, l] * x[m, j]
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
        for j in range(old_size_I):
            if Vsize_I[j] > 0:
                I[size_I] = I[j]
                size_I += 1

        n_iter += 1

    return n_iter if I.size == 0 else -1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _equal_column(uint8[:, :] mat_A, int col_A, uint8[:, :] mat_B, int col_B, int N_row):
    for i in range(N_row):
        if mat_A[i, col_A] != mat_B[i, col_B]:
            return 0

    return 1
