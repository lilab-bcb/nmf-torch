import torch

from typing import Union
from ._nmf_online_base import NMFOnlineBase


class NMFOnlineHALS(NMFOnlineBase):
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
        chunk_max_iter: int = 200,
        h_tol: float = 0.05,
        w_tol: float = 0.05,
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

        self._chunk_max_iter = chunk_max_iter
        self._h_tol = h_tol
        self._w_tol = w_tol
        self._zero = torch.tensor(0.0, dtype=self._tensor_dtype, device=self._device_type)


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

            for j in range(self._chunk_max_iter):
                cur_max = 0.0

                for k in range(self.k):
                    numer = xWT[:, k] - h @ WWT[:, k]
                    if self._l1_reg_H > 0.0:
                        numer -= self._l1_reg_H
                    if self._l2_reg_H > 0.0:
                        denom = WWT[k, k] + self._l2_reg_H
                        hvec = h[:, k] * (WWT[k, k] / denom) + numer / denom
                    else:
                        hvec = h[:, k] + numer / WWT[k, k]
                    if torch.isnan(hvec).sum() > 0:
                        hvec[:] = 0.0 # divide zero error: set hvec to 0
                    else:
                        hvec = hvec.maximum(self._zero)
                    cur_max = max(cur_max, torch.abs(h[:, k] - hvec).max())
                    h[:, k] = hvec
                
                if j + 1 < self._chunk_max_iter and cur_max / h.mean() < self._h_tol:
                            break

            # print(f"Block {i} update H iterates {j+1} iterations.")
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
            for j in range(self._chunk_max_iter):
                cur_max = 0.0

                for k in range(self.k):
                    numer = B[k, :] - A[k, :] @ self.W
                    if l1_reg_W > 0.0:
                        numer -= l1_reg_W
                    if l2_reg_W > 0.0:
                        denom = A[k, k] + l2_reg_W
                        w_new = self.W[k, :] * (A[k, k] / denom) + numer / denom
                    else:
                        w_new = self.W[k, :] + numer / A[k, k]
                    if torch.isnan(w_new).sum() > 0:
                        w_new[:] = 0.0 # divide zero error: set w_new to 0
                    else:
                        w_new = w_new.maximum(self._zero)
                    cur_max = max(cur_max, torch.abs(self.W[k, :] - w_new).max())
                    self.W[k, :] = w_new
                
                if j + 1 < self._chunk_max_iter and cur_max / self.W.mean() < self._w_tol:
                    break

            # print(f"Block {i} update W iterates {j+1} iterations.")
            i += self._chunk_size


    def _update_H(self):
        i = 0
        WWT = self.W @ self.W.T

        sum_h_err = torch.tensor(0.0, dtype=torch.double, device=self._device_type) # make sure sum_h_err is double to avoid summation errors
        while i < self.H.shape[0]:
            x = self.X[i:(i+self._chunk_size), :]
            h = self.H[i:(i+self._chunk_size), :]

            xWT = x @ self.W.T
            for j in range(self._chunk_max_iter):
                cur_max = 0.0

                for k in range(self.k):
                    numer = xWT[:, k] - h @ WWT[:, k]
                    if self._l1_reg_H > 0.0:
                        numer -= self._l1_reg_H
                    if self._l2_reg_H > 0.0:
                        denom = WWT[k, k] + self._l2_reg_H
                        hvec = h[:, k] * (WWT[k, k] / denom) + numer / denom
                    else:
                        hvec = h[:, k] + numer / WWT[k, k]
                    if torch.isnan(hvec).sum() > 0:
                        hvec[:] = 0.0 # divide zero error: set hvec to 0
                    else:
                        hvec = hvec.maximum(self._zero)
                    cur_max = max(cur_max, torch.abs(h[:, k] - hvec).max())
                    h[:, k] = hvec

                if j + 1 < self._chunk_max_iter and cur_max / h.mean() < self._h_tol:
                    break

            # print(f"Block {i} update H iterates {j+1} iterations.")

            hth = h.T @ h
            sum_h_err += self._h_err(h, hth, WWT, xWT)

            i += self._chunk_size

        return torch.sqrt(2.0 * (sum_h_err + self._X_SS_half + self._get_regularization_loss(self.W, self._l1_reg_W, self._l2_reg_W)))


    def fit(self, X):
        super().fit(X)
        assert self._beta==2, "Cannot perform online update when beta is not equal to 2!"

        # Online update.
        self._chunk_size = min(self.X.shape[0], self._chunk_size)

        l1_reg_W = self._l1_reg_W / self.X.shape[0]
        l2_reg_W = self._l2_reg_W / self.X.shape[0]

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

        print(f"Update H")
        self._cur_err = self._update_H()
        print(f"Final loss={self._cur_err}.")
