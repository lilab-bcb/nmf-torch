import torch

from ._inmf_base import INMFBase
from typing import List, Union

class INMFOnlineHALS(INMFBase):
    def __init__(
        self,
        n_components: int,
        lam: float = 5.,
        init: str = 'random',
        tol: float = 1e-4,
        random_state: int = 0,
        fp_precision: Union[str, torch.dtype] = 'float',
        device_type: str = 'cpu',
        max_pass: int = 10,
        chunk_size: int = 2000,
        w_max_iter: int = 200,
        v_max_iter: int = 50,
        h_max_iter: int = 50,
        w_tol: float = 1e-4,
        v_tol: float = 1e-4,
        h_tol: float = 1e-4,
    ):
        super().__init__(
            n_components=n_components,
            lam=lam,
            init=init,
            tol=tol,
            random_state=random_state,
            fp_precision=fp_precision,
            device_type=device_type,
        )

        self._max_pass = max_pass
        self._chunk_size = chunk_size
        self._w_max_iter = w_max_iter
        self._v_max_iter = v_max_iter
        self._h_max_iter = h_max_iter


    def _h_err(self, h, hth, WVWVT, xWVT, VVT):
        # Calculate L2 Loss (no sum of squares of X) for block h in trace format.
        res = self._trace(WVWVT + self._lambda * VVT, hth) if self._lambda > 0.0 else self._trace(WVWVT, hth)
        res -= 2.0 * self._trace(h, xWVT)
        return res


    def _v_err(self, A, B, WV, WVWVT, VVT):
        # Calculate L2 Loss (no sum of squares of X) for one batch in trace format.
        res = self._trace(WVWVT + self._lambda * VVT, A) if self._lambda > 0.0 else self._trace(WVWVT, A)
        res -= 2.0 * self._trace(B, WV)
        return res


    def _w_err(self, CW, E, D):
        res = self._trace(CW + 2.0 * E, self.W) - 2.0 * self._trace(D, self.W)
        return res


    def _update_one_pass(self):
        """
            A = sum hth; B = sum htx; for each batch
            C = sum of hth; D = sum of htx; E = sum of AV; for all batches
        """
        A = torch.zeros((self._n_components, self._n_components), dtype=self._tensor_dtype, device=self._device_type)
        B = torch.zeros((self._n_components, self._n_features), dtype=self._tensor_dtype, device=self._device_type)
        C = torch.zeros((self._n_components, self._n_components), dtype=self._tensor_dtype, device=self._device_type)
        D = torch.zeros((self._n_components, self._n_features), dtype=self._tensor_dtype, device=self._device_type)
        E = torch.zeros((self._n_components, self._n_features), dtype=self._tensor_dtype, device=self._device_type)

        batch_indices = torch.randperm(self._n_batches, device=self._device_type)
        for k in batch_indices:
            indices = torch.randperm(self.X[k].shape[0], device=self._device_type)

            # Block-wise update
            i = 0
            VVT = self.V[k] @ self.V[k].T if self._lambda > 0.0 else None
            A.fill_(0.0)
            B.fill_(0.0)
            while i < indices.shape[0]:
                idx = indices[i:(i+self._chunk_size)]
                x = self.X[k][idx, :]
                h = self.H[k][idx, :]

                # Update H
                WV = self.W + self.V[k]
                WVWVT = WV @ WV.T
                hth = h.T @ h
                xWVT = x @ WV.T

                cur_h_err = self._h_err(h, hth, WVWVT, xWVT, VVT)

                for j in range(self._h_max_iter):
                    prev_h_err = cur_h_err

                    for l in range(self._n_components):
                        if self._lambda > 0.0:
                            numer = xWVT[:, l] - h @ (WVWVT[:, l] + self._lambda * VVT[:, l])
                            denom = WVWVT[l, l] + self._lambda * VVT[l, l]
                        else:
                            numer = xWVT[:, l] - h @ WVWVT[:, l]
                            denom = WVWVT[l, l]    
                        hvec = h[:, l] + numer / denom
                        if torch.isnan(hvec).sum() > 0:
                            hvec[:] = 0.0 # divide zero error: set h_new to 0
                        else:
                            hvec = hvec.maximum(self._zero)
                        h[:, l] = hvec

                    hth = h.T @ h
                    cur_h_err = self._h_err(h, hth, WVWVT, xWVT, VVT)

                    if self._is_converged(prev_h_err, cur_h_err, prev_h_err):
                        break

                self.H[k][idx, :] = h

                # Update sufficient statistics for batch k
                A += hth
                htx = h.T @ x
                B += htx

                # Update V
                cur_v_err = self._v_err(A, B, WV, WVWVT, VVT)

                for j in range(self._v_max_iter):
                    prev_v_err = cur_v_err

                    for l in range(self._n_components):
                        numer = B[l, :] - A[l, :] @ (self.W + (1.0 + self._lambda) * self.V[k])
                        denom = (1.0 + self._lambda) * A[l, l]
                        v_new = self.V[k][l, :] + numer / denom
                        if torch.isnan(v_new).sum() > 0:
                            v_new[:] = 0.0 # divide zero error: set v_new to 0
                        else:
                            v_new = v_new.maximum(self._zero)
                        self.V[k][l, :] = v_new

                    WV = self.W + self.V[k]
                    WVWVT = WV @ WV.T
                    VVT = self.V[k] @ self.V[k].T if self._lambda > 0.0 else None
                    cur_v_err = self._v_err(A, B, WV, WVWVT, VVT)

                    if self._is_converged(prev_v_err, cur_v_err, prev_v_err):
                        break

                # Update sufficient statistics for all batches
                C += hth
                D += htx
                CW = C @ self.W
                E_new = E + A @ self.V[k]

                # Update W
                cur_w_err = self._w_err(CW, E_new, D)
                for j in range(self._w_max_iter):
                    prev_w_err = cur_w_err

                    for l in range(self._n_components):
                        w_new = self.W[l, :] + (D[l, :] - E_new[l, :] - CW[l, :]) / C[l, l]
                        if torch.isnan(w_new).sum() > 0:
                            w_new[:] = 0.0 # divide zero error: set w_new to 0
                        else:
                            w_new = w_new.maximum(self._zero)
                        self.W[l, :] = w_new
                    CW = C @ self.W
                    cur_w_err = self._w_err(CW, E_new, D)

                    if self._is_converged(prev_w_err, cur_w_err, prev_w_err):
                        break

                i += self._chunk_size
            E = E_new


    def _update_H_V(self):
        """
            Fix W, only update V and H
            A = sum hth; B = sum htx; for each batch
        """
        A = torch.zeros((self._n_components, self._n_components), dtype=self._tensor_dtype, device=self._device_type)
        B = torch.zeros((self._n_components, self._n_features), dtype=self._tensor_dtype, device=self._device_type)

        for k in range(self._n_batches):
            indices = torch.randperm(self.X[k].shape[0], device=self._device_type)

            # Block-wise update
            i = 0
            WV = self.W + self.V[k]
            WVWVT = WV @ WV.T
            VVT = self.V[k] @ self.V[k].T if self._lambda > 0.0 else None
            A.fill_(0.0)
            B.fill_(0.0)
            while i < indices.shape[0]:
                idx = indices[i:(i+self._chunk_size)]
                x = self.X[k][idx, :]
                h = self.H[k][idx, :]

                # Update H
                hth = h.T @ h
                xWVT = x @ WV.T

                cur_h_err = self._h_err(h, hth, WVWVT, xWVT, VVT)

                for j in range(self._h_max_iter):
                    prev_h_err = cur_h_err

                    for l in range(self._n_components):
                        if self._lambda > 0.0:
                            numer = xWVT[:, l] - h @ (WVWVT[:, l] + self._lambda * VVT[:, l])
                            denom = WVWVT[l, l] + self._lambda * VVT[l, l]
                        else:
                            numer = xWVT[:, l] - h @ WVWVT[:, l]
                            denom = WVWVT[l, l]    
                        hvec = h[:, l] + numer / denom
                        if torch.isnan(hvec).sum() > 0:
                            hvec[:] = 0.0 # divide zero error: set h_new to 0
                        else:
                            hvec = hvec.maximum(self._zero)
                        h[:, l] = hvec

                    hth = h.T @ h
                    cur_h_err = self._h_err(h, hth, WVWVT, xWVT, VVT)

                    if self._is_converged(prev_h_err, cur_h_err, prev_h_err):
                        break

                self.H[k][idx, :] = h

                # Update sufficient statistics for batch k
                A += hth
                htx = h.T @ x
                B += htx

                # Update V
                cur_v_err = self._v_err(A, B, WV, WVWVT, VVT)

                for j in range(self._v_max_iter):
                    prev_v_err = cur_v_err

                    for l in range(self._n_components):
                        numer = B[l, :] - A[l, :] @ (self.W + (1.0 + self._lambda) * self.V[k])
                        denom = (1.0 + self._lambda) * A[l, l]
                        v_new = self.V[k][l, :] + numer / denom
                        if torch.isnan(v_new).sum() > 0:
                            v_new[:] = 0.0 # divide zero error: set v_new to 0
                        else:
                            v_new = v_new.maximum(self._zero)
                        self.V[k][l, :] = v_new

                    WV = self.W + self.V[k]
                    WVWVT = WV @ WV.T
                    VVT = self.V[k] @ self.V[k].T if self._lambda > 0.0 else None
                    cur_v_err = self._v_err(A, B, WV, WVWVT, VVT)

                    if self._is_converged(prev_v_err, cur_v_err, prev_v_err):
                        break

                i += self._chunk_size


    def _update_H(self):
        """ Fix W and V, update H """
        sum_h_err = 0.0
        for k in range(self._n_batches):
            WV = self.W + self.V[k]
            WVWVT = WV @ WV.T
            VVT = self.V[k] @ self.V[k].T if self._lambda > 0.0 else None

            i = 0
            while i < self.H[k].shape[0]:
                x = self.X[k][i:(i+self._chunk_size), :]
                h = self.H[k][i:(i+self._chunk_size), :]

                # Update H
                hth = h.T @ h
                xWVT = x @ WV.T

                cur_h_err = self._h_err(h, hth, WVWVT, xWVT, VVT)

                for j in range(self._h_max_iter):
                    prev_h_err = cur_h_err

                    for l in range(self._n_components):
                        if self._lambda > 0.0:
                            numer = xWVT[:, l] - h @ (WVWVT[:, l] + self._lambda * VVT[:, l])
                            denom = WVWVT[l, l] + self._lambda * VVT[l, l]
                        else:
                            numer = xWVT[:, l] - h @ WVWVT[:, l]
                            denom = WVWVT[l, l]    
                        hvec = h[:, l] + numer / denom
                        if torch.isnan(hvec).sum() > 0:
                            hvec[:] = 0.0 # divide zero error: set h_new to 0
                        else:
                            hvec = hvec.maximum(self._zero)
                        h[:, l] = hvec

                    hth = h.T @ h
                    cur_h_err = self._h_err(h, hth, WVWVT, xWVT, VVT)

                    if self._is_converged(prev_h_err, cur_h_err, prev_h_err):
                        break

                sum_h_err += cur_h_err
                i += self._chunk_size

        return sum_h_err


    def fit(
        self,
        mats: List[torch.tensor],
    ):
        super().fit(mats)

        for i in range(self._max_pass):
            self._update_one_pass()
            self._update_H_V()
            H_err = self._update_H()

            self._cur_err = torch.sqrt(H_err + self._SSX)
            if self._is_converged(self._prev_err, self._cur_err, self._init_err, self._tol):
                self.num_iters = i + 1
                print(f"    Converged after {self.num_iters} pass(es).")
                return

            self._prev_err = self._cur_err

        self.num_iters = self._max_pass
        print(f"    Not converged after {self._max_pass} pass(es).")


    def fit_transform(
        self,
        mats: List[torch.tensor],
    ):
        self.fit(mats)
        return self.W