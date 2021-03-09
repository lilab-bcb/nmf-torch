import torch

from typing import Union
from ._nmf_base import NMFBase


class NMFBatch(NMFBase):
    def __init__(
        self,
        n_components: int,
        init,
        beta_loss: float,
        tol: float,
        random_state: int,
        alpha_W: float,
        l1_ratio_W: float,
        alpha_H: float,
        l1_ratio_H: float,
        fp_precision: Union[str, torch.dtype],
        device_type: str,
        max_iter: int,
    ):
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
        )

        self._max_iter = max_iter


    def _update_H(self):
        if self._beta == 2:
            H_factor_numer = self._XWT.clone()
            H_factor_denom = self.H @ self._WWT
        else:
            HW = self._get_HW()
            HW_pow = HW.pow(self._beta - 2)
            H_factor_numer = (self.X * HW_pow) @ self._W_t
            H_factor_denom = (HW_pow * HW) @ self._W_t

        self._add_regularization_terms(self.H, H_factor_numer, H_factor_denom, self._l1_reg_H, self._l2_reg_H)
        self._update_matrix(self.H, H_factor_numer, H_factor_denom)

        if self._beta == 2:
            self._H_t = self.H.T
            self._HTH = self._H_t @ self.H


    def _update_W(self):
        if self._beta == 2:
            W_factor_numer = self._H_t @ self.X
            W_factor_denom = self._HTH @ self.W
        else:
            H_t = self.H.T
            HW = self._get_HW()
            HW_pow = HW.pow(self._beta - 2)
            W_factor_numer = H_t @ (self.X * HW_pow)
            W_factor_denom = H_t @ (HW_pow * HW)

        self._add_regularization_terms(self.W, W_factor_numer, W_factor_denom, self._l1_reg_W, self._l2_reg_W)
        self._update_matrix(self.W, W_factor_numer, W_factor_denom)

        if self._beta == 2:
            self._W_t = self.W.T
            self._WWT = self.W @ self._W_t
            self._XWT = self.X @ self._W_t


    @torch.no_grad()
    def fit(self, X):
        super().fit(X)

        # Batch update.
        for i in range(self._max_iter):
            if (i + 1) % 10 == 0:
                self._cur_err = self._loss()
                if self._is_converged(self._prev_err, self._cur_err, self._init_err):
                    self.num_iters = i + 1
                    print(f"    Converged after {self.num_iters} iteration(s).")
                    break
                else:
                    self._prev_err = self._cur_err

            self._update_H()
            self._update_W()

            if i == self._max_iter - 1:
                self.num_iters = self._max_iter
                print(f"    Not converged after {self.num_iters} iteration(s).")


    def fit_transform(self, X):
        self.fit(X)
        return self.H
