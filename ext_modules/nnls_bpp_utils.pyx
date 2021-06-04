# cython: language_level=3
# distutils: language = c++

### NNLS Block principal pivoting: Kim and Park et al 2011.

import numpy as np
import torch

cimport cython

from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp.string cimport string
from cython.operator import dereference, postincrement

ctypedef unsigned char uint8


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _nnls_bpp(float[:, :] CTC, float[:, :] CTB, float[:, :] X, str device_type):
    # CTC = C.T @ C, CTB = C.T @ B, X.shape = (q, r)

    cdef Py_ssize_t i, j, k, l, m, uniq_idx
    cdef int q = CTB.shape[0]
    cdef int r = CTB.shape[1]
    cdef int max_iter = 5 * q

    CTC_L = torch.zeros((q, q), dtype=torch.float, device=device_type)
    CTB_L = torch.zeros((q, r), dtype=torch.float, device=device_type)
    cdef Py_ssize_t CTC_L_M, CTC_L_N, CTB_L_M, CTB_L_N

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

    cdef unordered_map[string, vector[int]] uniq_F = unordered_map[string, vector[int]]()
    cdef unordered_map[string, vector[int]].iterator it
    cdef string Fvec_str
    cdef int[:] indices = np.zeros((r,), dtype=np.int32)
    cdef uint8[:] Fvec = np.zeros((q,), dtype=np.bool_)
    cdef int[:] Ii = np.zeros((r,), dtype=np.int32)
    cdef uint8[:, :] V_I = np.zeros((q, r), dtype=np.bool_)
    cdef int[:] Vsize_I = np.zeros((r,), dtype=np.int32)
    cdef float[:, :] y = np.zeros((q, r), dtype=np.float32)

    cdef int col_idx, row_idx, nF, nG, size_uniq_F, uniq_flag, size_Ii
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
        uniq_F.clear()
        for j in range(size_I):
            Fvec_str = string(b' ', q)
            for i in range(q):
                Fvec_str[i] = b'1' if F[i, I[j]] else b'0'

            if uniq_F.count(Fvec_str) > 0:
                uniq_F[Fvec_str].push_back(j)
            else:
                uniq_F[Fvec_str] = vector[int](1, j)

        # Solve grouped normal equations
        it = uniq_F.begin()
        while it != uniq_F.end():
            for i in range(q):
                Fvec[i] = 1 if dereference(it).first[i] == b'1' else 0

            size_Ii = 0
            for i in range(dereference(it).second.size()):
                Ii[size_Ii] = I[dereference(it).second[i]]
                size_Ii += 1

            nF = 0
            for i in range(Fvec.size):
                nF += Fvec[i]
            nG = q - nF

            if nF > 0:
                # CTC_L = CTC[Fvec, Fvec]
                CTC_L_M = 0
                for i in range(q):
                    if Fvec[i]:
                        CTC_L_N = 0
                        for j in range(q):
                            if Fvec[j]:
                                CTC_L[CTC_L_M, CTC_L_N] = CTC[i, j]
                                CTC_L_N += 1
                        CTC_L_M += 1
                assert CTC_L_M == CTC_L_N, f"CTC_L of shape ({CTC_L_M}, {CTC_L_N}) is not square!"

                L = torch.cholesky(CTC_L[0:CTC_L_M, 0:CTC_L_N])

                # CTB_L = CTB[Fvec, Ii]
                CTB_L_M = 0
                for i in range(q):
                    if Fvec[i]:
                        CTB_L_N = 0
                        for j in range(size_Ii):
                            CTB_L[CTB_L_M, CTB_L_N] = CTB[i, Ii[j]]
                            CTB_L_N += 1
                        CTB_L_M += 1
                assert CTB_L_M==CTC_L_M and CTB_L_N==size_Ii, f"CTB_L has shape ({CTB_L_M}, {CTB_L_N}), but expect ({CTC_L_M}, {size_Ii})."

                x = torch.cholesky_solve(CTB_L[0:CTB_L_M, 0:CTB_L_N], L)

                k = 0
                for i in range(q):
                    if Fvec[i]:
                        for j in range(size_Ii):
                            X[i, Ii[j]] = x[k, j] if np.abs(x[k, j]) >= 1e-12 else 0.0
                            Y[i, Ii[j]] = 0.0
                        k += 1

            if nG > 0:
                if nF > 0:
                    # CTC_L = CTC[~Fvec, Fvec]
                    CTC_L_M = 0
                    for i in range(q):
                        if not Fvec[i]:
                            CTC_L_N = 0
                            for j in range(q):
                                if Fvec[j]:
                                    CTC_L[CTC_L_M, CTC_L_N] = CTC[i, j]
                                    CTC_L_N += 1
                            CTC_L_M += 1
                    assert CTC_L_M + CTC_L_N == q, "CTC_L has shape ({CTC_L_M}, {CTC_L_N}), but expect sum of dims = {q}!"
                    y_tensor = CTC_L[0:CTC_L_M, 0:CTC_L_N] @ x
                    y = y_tensor.cpu().numpy()

                k = 0
                for i in range(q):
                    if not Fvec[i]:
                        for j in range(size_Ii):
                            if nF <= 0:
                                y[k, j] = 0.0
                            y[k, j] -= CTB[i, Ii[j]]
                            if np.abs(y[k, j]) < 1e-12:
                                y[k, j] = 0.0
                            Y[i, Ii[j]] = y[k, j]
                            X[i, Ii[j]] = 0.0
                        k += 1

            postincrement(it)

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

    return n_iter if size_I == 0 else -1
