import torch

from ._nmf_base import NMFBase
from typing import Union


class NMFBatchBase(NMFBase):
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
            n_jobs=n_jobs,
        )

        self._max_iter = max_iter


    def _loss(self):
        res = torch.tensor(0.0, dtype=torch.double, device=self._device_type) # make sure res is double to avoid summation errors
        if self._beta == 2:
            res += self._trace(self._WWT, self._HTH) / 2.0 - self._trace(self.H, self._XWT) + self._X_SS_half
        elif self._beta == 0 or self._beta == 1:
            Y = self._get_HW()
            X_flat = self.X.flatten()
            Y_flat = Y.flatten()

            idx = X_flat > self._epsilon
            X_flat = X_flat[idx]
            Y_flat = Y_flat[idx]

            # Avoid division by zero
            Y_flat[Y_flat == 0] = self._epsilon

            x_div_y = X_flat / Y_flat
            if self._beta == 0:
                res += x_div_y.sum() - x_div_y.log().sum() - self.X.shape.numel()
            else:
                res += X_flat @ x_div_y.log() - X_flat.sum() + Y.sum()
        else:
            Y = self._get_HW()
            res += (torch.sum(self.X.pow(self._beta) - self._beta * self.X * Y.pow(self._beta - 1) + (self._beta - 1) * Y.pow(self._beta))) / (self._beta * (self._beta - 1))

        # Add regularization terms.
        res += self._get_regularization_loss(self.H, self._l1_reg_H, self._l2_reg_H)
        res += self._get_regularization_loss(self.W, self._l1_reg_W, self._l2_reg_W)

        return torch.sqrt(2.0 * res)


    def fit(self, X):
        super().fit(X)

        if self._beta == 2:
            self._WWT = self.W @ self.W.T
            self._HTH = self.H.T @ self.H
            self._XWT = self.X @ self.W.T

        self._init_err = self._loss()
        self._prev_err = self._init_err
