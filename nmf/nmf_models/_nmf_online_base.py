import torch

from ._nmf_base import NMFBase
from typing import Union


class NMFOnlineBase(NMFBase):
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
        max_pass: int = 20,
        chunk_size: int = 5000,
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
        )

        self._max_pass = max_pass
        self._chunk_size = chunk_size


    def _h_err(self, h, hth, WWT, xWT):
        # Forbenious-norm^2 in trace format (No X)
        res = self._trace(WWT, hth) / 2.0 - self._trace(h, xWT)
        # Add regularization terms if needed
        if self._l1_reg_H > 0.0:
            res += self._l1_reg_H * h.norm(p=1)
        if self._l2_reg_H > 0.0:
            res += self._l2_reg_H * torch.trace(hth) / 2.0
        return res

    def _loss(self):
        """ calculate loss online by passing through all data"""
        i = 0
        WWT = self.W @ self.W.T

        sum_h_err = torch.tensor(0.0, dtype=torch.double, device=self._device_type) # make sure sum_h_err is double to avoid summation errors
        while i < self.H.shape[0]:
            x = self.X[i:(i+self._chunk_size), :]
            h = self.H[i:(i+self._chunk_size), :]
            xWT = x @ self.W.T
            hth = h.T @ h
            sum_h_err += self._h_err(h, hth, WWT, xWT)
            i += self._chunk_size

        return torch.sqrt(2.0 * (sum_h_err + self._X_SS_half + self._get_regularization_loss(self.W, self._l1_reg_W, self._l2_reg_W)))


    def fit(self, X):
        super().fit(X)

        self._init_err = self._loss()
        self._prev_err = self._init_err
