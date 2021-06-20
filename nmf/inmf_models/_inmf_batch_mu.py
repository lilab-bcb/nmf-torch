import torch

from ._inmf_batch_base import INMFBatchBase
from typing import List


class INMFBatchMU(INMFBatchBase):
    def _update_matrix(self, mat, numer, denom):
        mat *= (numer / denom)
        mat[denom < self._epsilon] = 0.0


    def _update_H_V_W(self):
        W_numer = torch.zeros_like(self.W)
        W_denom = torch.zeros_like(self.W)
        # Update Hs and Vs and calculate partials sum for updating W
        for k in range(self._n_batches):
            # Update H[k]
            H_numer = self._XWVT[k]
            H_denom = self.H[k] @ (self._WVWVT[k] + self._lambda * self._VVT[k]) if self._lambda > 0.0 else self.H[k] @ self._WVWVT[k]
            self._update_matrix(self.H[k], H_numer, H_denom)
            # Cache HTH
            self._HTH[k] = self.H[k].T @ self.H[k]

            # Update V[k]
            V_numer = self.H[k].T @ self.X[k]
            V_denom = self._HTH[k] @ (self.W + (1.0 + self._lambda) * self.V[k])
            self._update_matrix(self.V[k], V_numer, V_denom)
            # Cache VVT
            if self._lambda > 0.0:
                self._VVT[k] = self.V[k] @ self.V[k].T

            # Update W numer and denomer
            W_numer += V_numer
            W_denom += self._HTH[k] @ (self.W + self.V[k])
        # Update W
        self._update_matrix(self.W, W_numer, W_denom)
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
