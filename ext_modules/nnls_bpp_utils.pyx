# cython: language_level=3
# distutils: language = c++

### NNLS Block principal pivoting: Kim and Park et al 2011.

import numpy as np
import torch

cimport cython

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from cython.operator import dereference as deref
from cython.operator import postincrement as pinc

ctypedef unsigned char uint8

ctypedef fused array_type:
    float
    double


cdef inline array_type _filter(array_type number, array_type tol):
    if number > -tol and number < tol:
        return 0.0
    return number


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _nnls_bpp(array_type[:, :] CTC, array_type[:, :] CTB, array_type[:, :] X, str device_type):
    # CTC = C.T @ C, CTB = C.T @ B, X.shape = (q, r)
    numpy_type = np.float32 if array_type is float else np.float64
    torch_type = torch.float if array_type is float else torch.double
    cdef array_type tol = 1e-6 if array_type is float else 1e-12

    cdef Py_ssize_t i, j, col_idx, row_idx
    cdef Py_ssize_t n_iter, fvsize, gvsize, uqsize, size_I, pos

    cdef int q = CTB.shape[0]
    cdef int r = CTB.shape[1]
    cdef int max_iter = 5 * q

    cdef int backup_cap = 3 # maximum back up tries
    cdef int[:] alpha = np.full(r, backup_cap, dtype=np.int32)  # cap on back up rule
    cdef int[:] beta = np.full(r, q+1, dtype=np.int32)  # number of infeasible variables

    ### Initialization, setting G = 1-q
    cdef array_type[:, :] Y = np.zeros_like(CTB, dtype=numpy_type)
    cdef uint8[:, :] V = np.zeros((q, r), dtype=np.bool_)
    cdef int[:] Vsize = np.zeros((r,), dtype=np.int32)

    cdef vector[int] I

    cdef uint8[:, :] F = np.zeros((q, r), dtype=np.bool_)  # y_F = 0, G = ~F, x_G = 0

    CTC_L_tensor = torch.zeros((q, q), dtype=torch_type, device='cpu')
    cdef array_type[:, :] CTC_L = CTC_L_tensor.numpy()
    CTB_L_tensor = torch.zeros((q, r), dtype=torch_type, device='cpu')
    cdef array_type[:, :] CTB_L = CTB_L_tensor.numpy()

    cdef array_type[:, :] x
    cdef array_type[:, :] y

    cdef unordered_map[string, vector[int]] uniq_F
    cdef unordered_map[string, vector[int]].iterator it

    cdef string Fvec_str
    Fvec_str.resize(q, b' ')

    cdef vector[int] Fvec
    cdef vector[int] Gvec


    I.clear()
    for j in range(r):
        for i in range(q):
            X[i, j] = 0.0
            Y[i, j] = -CTB[i, j]
            V[i, j] = Y[i, j] < 0
            Vsize[j] += V[i, j]

        if Vsize[j] > 0:
            I.push_back(j)

    n_iter = 0
    while I.size() > 0 and n_iter < max_iter:
        # Split indices in I into 3 cases:
        for col_idx in I:
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
                F[row_idx, col_idx] ^= 1

        # Get unique F columns with indices mapping back to F.
        uniq_F.clear()
        for col_idx in I:
            for i in range(q):
                Fvec_str[i] = b'1' if F[i, col_idx] else b'0'

            it = uniq_F.find(Fvec_str)
            if it != uniq_F.end():
                deref(it).second.push_back(col_idx)
            else:
                uniq_F[Fvec_str] = vector[int](1, col_idx)

        # Solve grouped normal equations
        it = uniq_F.begin()
        while it != uniq_F.end():
            Fvec.clear()
            Gvec.clear()
            for i in range(q):
                if deref(it).first[i] == b'1':
                    Fvec.push_back(i)
                else:
                    Gvec.push_back(i)

            fvsize = Fvec.size()
            gvsize = Gvec.size()
            uqsize = deref(it).second.size()

            if fvsize > 0:
                # CTC_L = CTC[Fvec, Fvec]
                for i in range(fvsize):
                    for j in range(fvsize):
                        CTC_L[i, j] = CTC[Fvec[i], Fvec[j]]
                L_tensor = torch.cholesky(CTC_L_tensor[0:fvsize, 0:fvsize]) if device_type == 'cpu' else torch.cholesky(CTC_L_tensor[0:fvsize, 0:fvsize].cuda())
                # CTB_L = CTB[Fvec, Ii]
                for i in range(fvsize):
                    for j in range(uqsize):
                        CTB_L[i, j] = CTB[Fvec[i], deref(it).second[j]]
                x_tensor = torch.cholesky_solve(CTB_L_tensor[0:fvsize, 0:uqsize], L_tensor) if device_type == 'cpu' else torch.cholesky_solve(CTB_L_tensor[0:fvsize, 0:uqsize].cuda(), L_tensor)
                x = x_tensor.cpu().numpy()
                # clean up
                for i in range(fvsize):
                    for j in range(uqsize):
                        X[Fvec[i], deref(it).second[j]] = _filter(x[i, j], tol)
                        Y[Fvec[i], deref(it).second[j]] = 0.0
            
            if gvsize > 0:
                if fvsize > 0:
                    # CTC_L = CTC[~Fvec, Fvec]
                    for i in range(gvsize):
                        for j in range(fvsize):
                            CTC_L[i, j] = CTC[Gvec[i], Fvec[j]]
                    y_tensor = (CTC_L_tensor[0:gvsize, 0:fvsize] @ x_tensor) if device_type == 'cpu' else (CTC_L_tensor[0:gvsize, 0:fvsize].cuda() @ x_tensor)
                    y = y_tensor.cpu().numpy()

                    for i in range(gvsize):
                        row_idx = Gvec[i]
                        for j in range(uqsize):
                            col_idx = deref(it).second[j]
                            Y[row_idx, col_idx] = _filter(y[i,j] - CTB[row_idx, col_idx], tol)
                            X[row_idx, col_idx] = 0.0
                else:
                    for i in range(gvsize):
                        row_idx = Gvec[i]
                        for j in range(uqsize):
                            col_idx = deref(it).second[j]
                            Y[row_idx, col_idx] = _filter(-CTB[row_idx, col_idx], tol)
                            X[row_idx, col_idx] = 0.0

            pinc(it)

        size_I = 0
        for j in range(I.size()):
            pos = I[j]
            Vsize[pos] = 0
            for i in range(q):
                V[i, pos] = ((X[i, pos] < 0.0) & F[i, pos]) | ((Y[i, pos] < 0.0) & (F[i, pos] == 0))
                Vsize[pos] += V[i, pos]
            if Vsize[pos] > 0:
                I[size_I] = pos
                size_I += 1
        I.resize(size_I)

        n_iter += 1

    return n_iter if I.size() == 0 else -1
