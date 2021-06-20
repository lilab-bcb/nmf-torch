import torch

from typing import Union
from ._nmf_batch_base import NMFBatchBase


class NMFBatchHALS(NMFBatchBase):
    def __init__(
        self,
        n_components: int,
        init: str,
        beta_loss: float,
        tol: float,
        random_state: int,
        alpha_W: float,
        l1_ratio_W: float,
        alpha_H: float,
        l1_ratio_H: float,
        fp_precision: Union[str, torch.dtype],
        device_type: str,
        n_jobs: int = -1,
        max_iter: int = 500,
        hals_tol: float = 0.05,
        hals_max_iter: int = 200,
    ):
        assert beta_loss == 2.0 # only work for F norm for now

        super().__init__(
            n_components=n_components,
            init=init,
            beta_loss=beta_loss,
            tol=tol,
            random_state=random_state,
            alpha_W=alpha_W,
            l1_ratio_W=l1_ratio_W,
            alpha_H=alpha_H,
            l1_ratio_H=l1_ratio_H,
            fp_precision=fp_precision,
            device_type=device_type,
            n_jobs=n_jobs,
            max_iter=max_iter,
        )

        self._zero = torch.tensor(0.0, dtype=self._tensor_dtype, device=self._device_type)
        self._hals_tol = hals_tol
        self._hals_max_iter = hals_max_iter


    def _update_H(self):
        for i in range(self._hals_max_iter):
            cur_max = 0.0    
            for k in range(self.k):
                numer = self._XWT[:, k] - self.H @ self._WWT[:, k]
                if self._l1_reg_H > 0.0:
                    numer -= self._l1_reg_H
                if self._l2_reg_H > 0.0:
                    denom = self._WWT[k, k] + self._l2_reg_H
                    h_new = self.H[:, k] * (self._WWT[k, k] / denom) + numer / denom
                else:
                    h_new = self.H[:, k] + numer / self._WWT[k, k]
                if torch.isnan(h_new).sum() > 0:
                    h_new[:] = 0.0 # divide zero error: set h_new to 0
                else:
                    h_new = h_new.maximum(self._zero)
                cur_max = max(cur_max, torch.abs(self.H[:, k] - h_new).max())
                self.H[:, k] = h_new
            if i + 1 < self._hals_max_iter and cur_max / self.H.mean() < self._hals_tol:
                break

        self._HTH = self.H.T @ self.H


    def _update_W(self):
        HTX = self.H.T @ self.X
        for i in range(self._hals_max_iter):
            cur_max = 0.0
            for k in range(self.k):
                numer = HTX[k, :] - self._HTH[k, :] @ self.W
                if self._l1_reg_W > 0.0:
                    numer -= self._l1_reg_W
                if self._l2_reg_W > 0.0:
                    denom = self._HTH[k, k] + self._l2_reg_W
                    w_new = self.W[k, :] * (self._HTH[k, k] / denom) + numer / denom
                else:
                    w_new = self.W[k, :] + numer / self._HTH[k, k]
                if torch.isnan(w_new).sum() > 0:
                    w_new[:] = 0.0 # divide zero error: set w_new to 0
                else:
                    w_new = w_new.maximum(self._zero)
                cur_max = max(cur_max, torch.abs(self.W[k, :] - w_new).max())
                self.W[k, :] = w_new
            if i + 1 < self._hals_max_iter and cur_max / self.W.mean() < self._hals_tol:
                break

        self._WWT = self.W @ self.W.T
        self._XWT = self.X @ self.W.T


    def fit(self, X):
        super().fit(X)

        # Batch update.
        for i in range(self._max_iter):
            self._update_H()
            self._update_W()

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
