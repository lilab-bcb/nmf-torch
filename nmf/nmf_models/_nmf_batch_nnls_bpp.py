import torch

from ._nmf_batch_base import NMFBatchBase
from ..utils import nnls_bpp
from typing import Union


class NMFBatchNnlsBpp(NMFBatchBase):
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

        if self._l2_reg_H > 0.0:
            self._l2_H_I = torch.eye(self.k, dtype=self._tensor_dtype, device=self._device_type) * self._l2_reg_H
        if self._l2_reg_W > 0.0:
            self._l2_W_I = torch.eye(self.k, dtype=self._tensor_dtype, device=self._device_type) * self._l2_reg_W


    def _get_regularization_loss(self, mat, l1_reg, l2_reg):
        res = 0.0
        if l1_reg > 0:
            dim = 0 if mat.shape[0] == self.k else 1
            res += l1_reg * mat.norm(p=1, dim=dim).norm(p=2)**2
        if l2_reg > 0:
            res += l2_reg * mat.norm(p=2)**2 / 2
        return res


    def _update_H(self):
        if self._l1_reg_H == 0.0 and self._l2_reg_H == 0.0:
            n_iter = nnls_bpp(self._WWT, self._XWT.T, self.H.T, self._device_type)
        else:
            CTC = self._WWT.clone()
            if self._l1_reg_H > 0.0:
                CTC += 2.0 * self._l1_reg_H
            if self._l2_reg_H > 0.0:
                CTC += self._l2_H_I
            n_iter = nnls_bpp(CTC, self._XWT.T, self.H.T, self._device_type)
        # print(f"H n_iter={n_iter}.")
        self._HTH = self.H.T @ self.H


    def _update_W(self):
        HTX = self.H.T @ self.X
        if self._l1_reg_W == 0.0 and self._l2_reg_W == 0.0:
            n_iter = nnls_bpp(self._HTH, HTX, self.W, self._device_type)
        else:
            CTC = self._HTH.clone()
            if self._l1_reg_W > 0.0:
                CTC += 2.0 * self._l1_reg_W
            if self._l2_reg_W > 0.0:
                CTC += self._l2_W_I
            n_iter = nnls_bpp(CTC, HTX, self.W, self._device_type)
        # print(f"W n_iter={n_iter}.")
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
