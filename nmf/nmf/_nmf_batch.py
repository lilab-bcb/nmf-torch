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
        update_method: str,
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

        if update_method == 'hals':
            if beta_loss != 2.0:
                print("HALS only supports beta loss = 2.0. Switch to MU update method.")
                update_method = 'mu'
        self._update_method = update_method

        self._max_iter = max_iter


    def _update_H_mu(self):
        if self._beta == 2:
            H_factor_numer = self._XWT.clone()
            H_factor_denom = self.H @ self._WWT
        else:
            HW = self._get_HW()
            HW_pow = HW.pow(self._beta - 2)
            H_factor_numer = (self.X * HW_pow) @ self.W.T
            H_factor_denom = (HW_pow * HW) @ self.W.T

        self._add_regularization_terms(self.H, H_factor_numer, H_factor_denom, self._l1_reg_H, self._l2_reg_H)
        self._update_matrix(self.H, H_factor_numer, H_factor_denom)

        if self._beta == 2:
            self._HTH = self.H.T @ self.H


    def _update_W_mu(self):
        if self._beta == 2:
            W_factor_numer = self.H.T @ self.X
            W_factor_denom = self._HTH @ self.W
        else:
            HW = self._get_HW()
            HW_pow = HW.pow(self._beta - 2)
            W_factor_numer = self.H.T @ (self.X * HW_pow)
            W_factor_denom = self.H.T @ (HW_pow * HW)

        self._add_regularization_terms(self.W, W_factor_numer, W_factor_denom, self._l1_reg_W, self._l2_reg_W)
        self._update_matrix(self.W, W_factor_numer, W_factor_denom)

        if self._beta == 2:
            self._WWT = self.W @ self.W.T
            self._XWT = self.X @ self.W.T


    def _update_H_hals(self):
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
            self.H[:, k] = h_new

        self._HTH = self.H.T @ self.H


    def _update_W_hals(self):
        HTX = self.H.T @ self.X
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
            self.W[k, :] = w_new

        self._WWT = self.W @ self.W.T
        self._XWT = self.X @ self.W.T


    def _update_H(self):
        if self._update_method == 'mu':
            self._update_H_mu()
        else:
            self._update_H_hals()


    def _update_W(self):
        if self._update_method == 'mu':
            self._update_W_mu()
        else:
            self._update_W_hals()


    @torch.no_grad()
    def fit(self, X):
        super().fit(X)

        # Batch update.
        for i in range(self._max_iter):
            self._update_H()
            self._update_W()

            if (i + 1) % 10 == 0:
                self._cur_err = self._loss()
                if self._is_converged(self._prev_err, self._cur_err, self._init_err):
                    self.num_iters = i + 1
                    print(f"    Converged after {self.num_iters} iteration(s).")
                    return

                self._prev_err = self._cur_err

        self.num_iters = self._max_iter
        print(f"    Not converged after {self.num_iters} iteration(s).")


    def fit_transform(self, X):
        self.fit(X)
        return self.H
