import torch

from ._nmf_online_base import NMFOnlineBase
from ..utils import nnls_bpp
from typing import Union


class NMFOnlineNnlsBpp(NMFOnlineBase):
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
            max_pass=max_pass,
            chunk_size=chunk_size,
        )

        if self._l2_reg_H > 0.0:
            self._l2_H_I = torch.eye(self.k, dtype=self._tensor_dtype, device=self._device_type) * self._l2_reg_H
        if self._l2_reg_W > 0.0:
            self._l2_W_I = torch.eye(self.k, dtype=self._tensor_dtype, device=self._device_type)


    def _get_regularization_loss(self, mat, l1_reg, l2_reg):
        res = 0.0
        if l1_reg > 0:
            dim = 0 if mat.shape[0] == self.k else 1
            res += l1_reg * mat.norm(p=1, dim=dim).norm(p=2)**2
        if l2_reg > 0:
            res += l2_reg * mat.norm(p=2)**2 / 2
        return res


    def _h_err(self, h, hth, WWT, xWT):
        # Forbenious-norm^2 in trace format (No X)
        res = self._trace(WWT, hth) / 2.0 - self._trace(h, xWT)
        # Add regularization terms if needed
        if self._l1_reg_H > 0.0:
            res += self._l1_reg_H * h.norm(p=1, dim=1).norm(p=2)**2
        if self._l2_reg_H > 0.0:
            res += self._l2_reg_H * torch.trace(hth) / 2.0
        return res


    def _update_one_pass(self, l1_reg_W, l2_reg_W):
        indices = torch.randperm(self.X.shape[0], device=self._device_type)
        A = torch.zeros((self.k, self.k), dtype=self._tensor_dtype, device=self._device_type)
        B = torch.zeros((self.k, self.X.shape[1]), dtype=self._tensor_dtype, device=self._device_type)

        i = 0
        num_processed = 0
        while i < indices.shape[0]:
            idx = indices[i:(i+self._chunk_size)]
            cur_chunksize = idx.shape[0]
            x = self.X[idx, :]
            h = self.H[idx, :]

            # Online update H.
            WWT = self.W @ self.W.T
            xWT = x @ self.W.T

            if self._l1_reg_H == 0.0 and self._l2_reg_H == 0.0:
                n_iter = nnls_bpp(WWT, xWT.T, h.T, self._device_type)
            else:
                CTC = WWT.clone()
                if self._l1_reg_H > 0.0:
                    CTC += 2.0 * self._l1_reg_H
                if self._l2_reg_H > 0.0:
                    CTC += self._l2_H_I
                n_iter = nnls_bpp(CTC, xWT.T, h.T, self._device_type)
            # print(f"Block {i} update H iterates {n_iter} iterations.")
            self.H[idx, :] = h

            # Update sufficient statistics A and B.
            num_after = num_processed + cur_chunksize

            A *= num_processed
            A += h.T @ h
            A /= num_after

            B *= num_processed
            B += h.T @ x
            B /= num_after

            num_processed = num_after

            # Online update W.
            if l1_reg_W == 0.0 and l2_reg_W == 0.0:
                n_iter = nnls_bpp(A, B, self.W, self._device_type)
            else:
                CTC = A.clone()
                if l1_reg_W > 0.0:
                    CTC += 2.0 * l1_reg_W
                if l2_reg_W > 0.0:
                    CTC += self._l2_W_I
                n_iter = nnls_bpp(CTC, B, self.W, self._device_type)
            # print(f"Block {i} update W iterates {n_iter} iterations.")

            i += self._chunk_size


    def _update_H(self):
        i = 0
        WWT = self.W @ self.W.T

        sum_h_err = torch.tensor(0.0, dtype=torch.double, device=self._device_type) # make sure sum_h_err is double to avoid summation errors
        while i < self.H.shape[0]:
            x = self.X[i:(i+self._chunk_size), :]
            h = self.H[i:(i+self._chunk_size), :]

            xWT = x @ self.W.T

            if self._l1_reg_H == 0.0 and self._l2_reg_H == 0.0:
                n_iter = nnls_bpp(WWT, xWT.T, h.T, self._device_type)
            else:
                CTC = WWT.clone()
                if self._l1_reg_H > 0.0:
                    CTC += 2.0 * self._l1_reg_H
                if self._l2_reg_H > 0.0:
                    CTC += self._l2_H_I
                n_iter = nnls_bpp(CTC, xWT.T, h.T, self._device_type)
            # print(f"Block {i} update H iterates {n_iter} iterations.")

            hth = h.T @ h
            sum_h_err += self._h_err(h, hth, WWT, xWT)

            i += self._chunk_size

        return torch.sqrt(2.0 * (sum_h_err + self._X_SS_half + self._get_regularization_loss(self.W, self._l1_reg_W, self._l2_reg_W)))


    def fit(self, X):
        super().fit(X)
        assert self._beta==2, "Cannot perform online update when beta not equal to 2!"

        # Online update.
        self._chunk_size = min(self.X.shape[0], self._chunk_size)

        l1_reg_W = self._l1_reg_W / self.X.shape[0]
        l2_reg_W = self._l2_reg_W / self.X.shape[0]
        if l2_reg_W > 0.0:
            self._l2_W_I *= l2_reg_W

        self.num_iters = -1
        for i in range(self._max_pass):
            self._update_one_pass(l1_reg_W, l2_reg_W)
            self._cur_err = self._loss()
            print(f"Pass {i+1}, loss={self._cur_err}.")

            if self._is_converged(self._prev_err, self._cur_err, self._init_err):
                self.num_iters = i + 1
                print(f"    Converged after {self.num_iters} pass(es).")
                break

            self._prev_err = self._cur_err

        if self.num_iters < 0:
            self.num_iters = self._max_pass
            print(f"    Not converged after {self._max_pass} pass(es).")

        print("Update H")
        self._cur_err = self._update_H()
        print(f"Final loss={self._cur_err}.")
