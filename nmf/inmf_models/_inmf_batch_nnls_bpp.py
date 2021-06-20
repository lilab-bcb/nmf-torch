import torch

from ._inmf_batch_base import INMFBatchBase
from ..utils import nnls_bpp
from typing import List


class INMFBatchNnlsBpp(INMFBatchBase):
    def _update_H_V_W(self):
        W_numer = torch.zeros_like(self.W)
        W_denom = torch.zeros_like(self._HTH[0])
        # Update Hs and Vs and calculate terms for updating W
        for k in range(self._n_batches):
            # Update H[k]
            if self._lambda > 0.0:
                n_iter = nnls_bpp(self._WVWVT[k] + self._lambda * self._VVT[k], self._XWVT[k].T, self.H[k].T, self._device_type)
            else:
                n_iter = nnls_bpp(self._WVWVT[k], self._XWVT[k].T, self.H[k].T, self._device_type)
            # print(f"Batch {k} H n_iter={n_iter}.")
            # Cache HTH
            self._HTH[k] = self.H[k].T @ self.H[k]

            # Update V[k]
            HTX = self.H[k].T @ self.X[k]
            n_iter = nnls_bpp(self._HTH[k] * (1.0 + self._lambda), HTX - self._HTH[k] @ self.W, self.V[k], self._device_type)
            # print(f"Batch {k} V n_iter={n_iter}.")
            # Cache VVT
            if self._lambda > 0.0:
                self._VVT[k] = self.V[k] @ self.V[k].T

            # Update W numer and denomer
            W_numer += (HTX - self._HTH[k] @ self.V[k])
            W_denom += self._HTH[k]

        # Update W
        n_iter = nnls_bpp(W_denom, W_numer, self.W, self._device_type)
        # print(f"W n_iter={n_iter}.")
        # Cache WVWVT and XWVT
        for k in range(self._n_batches):
            WV = self.W + self.V[k]
            self._WVWVT[k] = WV @ WV.T
            self._XWVT[k] = self.X[k] @ WV.T


    def fit(
        self,
        mats: List[torch.tensor],
    ):
        super().fit(mats)

        # Batch update
        for i in range(self._max_iter):
            self._update_H_V_W()

            if (i + 1) % 10 == 0:
                self._cur_err = self._loss()
                print(f" niter={i+1}, loss={self._cur_err}.")
                if self._is_converged(self._prev_err, self._cur_err, self._init_err):
                    self.num_iters = i + 1
                    print(f"    Converged after {self.num_iters} iteration(s).")
                    return

                self._prev_err = self._cur_err

        self.num_iters = self._max_iter
        print(f"    Not converged after {self.num_iters} iteration(s).")
