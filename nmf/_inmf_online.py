import torch

from ._inmf_base import INMFBase
from typing import List, Union

class INMFOnline(INMFBase):
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
        h_max_iter: int = 50,
    ):
        super.__init__(
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
        self._h_max_iter = h_max_iter


    def _h_err(self, h, WVWVT, xWVT, VVT=None, cache_hth=None):
        hth = h.T @ h
        if cache_hth is not None:
            cache_hth[0] = hth
        # Forbenious-norm^2 in trace format (No X)
        res = torch.trace(WVWVT @ hth) - 2.0 * torch.trace(h.T @ xWVT)
        # Add regularization terms if needed
        if self._lambda > 0.0:
            res += self._lambda * torch.trace(hth @ VVT)
        return res


    def _WV_err(self, A, B, k, WV=None, WVWVT=None, VVT=None):
        if WV is None:
            WV = self.W + self.V[k]
            WVWVT = WV @ WV.T
            VVT = self.V[k] @ self.V[k].T

        res = torch.trace(WVWVT @ A) - 2.0 * torch.trace(B @ WV.T)
        # Add regularization terms if needed
        if self._lambda > 0:
            res += self._labmda * torch.trace(A @ VVT)
        return res


    def _update_one_pass(self):
        W_factor_numer = torch.zeros((self._n_components, self._n_features), dtype=self._tensor_dtype, device=self._device_type)
        A = [torch.zeros((self._n_components, self._n_components), dtype=self._tensor_dtype, device=self._device_type) for _ in range(self._n_batches)]
        B = [torch.zeros((self._n_components, self._n_features), dtype=self._tensor_dtype, device=self._device_type) for _ in range(self._n_batches)]
        # C is sum of B
        C = torch.zeros_like(B[0])
        # D is sum of AV
        D = torch.zeros_like(B[0])
        # E is sum of A
        E = torch.zeros_like(A[0])

        batch_indices = torch.randperm(self._n_batches, device=self._device_type)
        for k in batch_indices:
            indices = torch.randperm(self.X[k].shape[0], device=self._device_type)

            # Block-wise update.
            i = 0
            num_processed = 0
            while i < indices.shape[0]:
                idx = indices[i:(i+self._chunk_size)]
                cur_chunksize = idx.shape[0]
                x = self.X[k][idx, :]
                h = self.H[k][idx, :]

                # Update H.
                WV = self.W + self.V[k]
                WVWVT = WV @ WV.T
                xWVT = x @ WV.T
                VVT = self.V @ self.V.T

                h_factor_numer = xWVT

                cache_hth = [None]
                cur_h_err = self._h_err(h, WVWVT, xWVT, VVT, cache_hth)

                for j in range(self._h_max_iter):
                    prev_h_err = cur_h_err

                    h_factor_denom = h @ WVWVT
                    if self._lambda:
                        h_factor_denom += self._lambda * (h @ VVT)
                    self._update_matrix(h, h_factor_numer, h_factor_denom)
                    cur_h_err = self._h_err(h, WVWVT, xWVT, VVT, cache_hth)

                    if self._is_converged(prev_h_err, cur_h_err, prev_h_err):
                        break

                self.H[k][idx, :] = h

                # Update sufficient statistics A and B.
                hth = cache_hth[0]
                num_after = num_processed + cur_chunksize

                A[k] *= num_processed
                A[k] += hth
                A[k] /= num_after

                B[k] *= num_processed
                B[k] += h.T @ x
                B[k] /= num_after

                num_processed = num_after

                # Update V and W.
                V_factor_numer = B[k]
                W_factor_numer = C + B[k]

                cur_WV_err = self._WV_err(A[k], B[k], k, WV, WVWVT, VVT)

                for j in range(self._w_max_iter):
                    prev_WV_err = cur_WV_err

                    V_factor_denom = A[k] @ (self.W + (1 + self._lambda) * self.V[k])
                    self._update_matrix(self.V[k], V_factor_numer, V_factor_denom)

                    W_factor_denom = D + self.W @ E + A[k] @ (self.W + self.V[k])
                    self._update_matrix(self.W, W_factor_numer, W_factor_denom)

                    cur_WV_err = self._WV_err(A[k], B[k], k)

                    if self._is_converged(prev_WV_err, cur_WV_err, prev_WV_err):
                        break

                # Update statistics of W.
                C += B[k]
                D += A[k] @ self.V[k]
                E += A[k]

                i += self._chunk_size



    def _update_H(self):
        for k in range(self._n_batches):
            WV = self.W @ self.V[k]
            WVWVT = WV @ WV.T

            i = 0
            sum_h_err = 0.
            while i < self.H.shape[0]:
                x = self.X[i:(i+self._chunk_size), :]
                h = self.H[i:(i+self._chunk_size), :]

                xWVT = x @ WV.T
                h_factor_numer = xWVT

                VVT = None
                if self._lambda > 0:
                    VVT = self.V[k] @ self.V[k].T

                cur_h_err = self._h_err(h, WVWVT, xWVT, VVT)

                for j in range(self._h_max_iter):
                    prev_h_err = cur_h_err

                    h_factor_denom = h @ WVWVT
                    if self._lambda > 0:
                        h_factor_denom += self._lambda * (h @ VVT)
                    self._update_matrix(h, h_factor_numer, h_factor_denom)
                    cur_h_err = self._h_err(h, WVWVT, xWVT, VVT)

                    if self._is_converged(prev_h_err, cur_h_err, prev_h_err):
                        break

                sum_h_err += cur_h_err
                i += self._chunk_size


    def fit(
        self,
        mats: List[torch.tensor],
    ):
        super().fit(mats)

        # Adjust chunk size if needed.
        x_min = self.X[0].shape[0]
        for k in range(1, self._n_batches):
            if x_min < self.X[k].shape[0]:
                x_min = self.X[k].shape[0]
        if self._chunk_size > x_min:
            print(f"Warning: The chunk size you set {self._chunk_size} is larger than number of cells in the smallest sample {x_min}. Set chunk size to {x_min} instead!")
            self._chunk_size = x_min

        for i in range(self._max_pass):
            self._update_one_pass()

            # Update H.
            H_err = self._update_H()

            self._cur_err = torch.sqrt(H_err + self._SSX)
            if self._is_converged(self._prev_err, self._cur_err, self._init_err):
                self.num_iters = i + 1
                print(f"    Converged after {self.num_iters} pass(es).")
                return

        self.num_iters = self._max_pass
        print(f"    Not converged after {self._max_pass} pass(es).")


    def fit_transform(
        self,
        mats: List[torch.tensor],
    ):
        self.fit(mats)
        return self.W
