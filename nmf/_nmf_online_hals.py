import torch

from typing import Union
from ._nmf_base import NMFBase


class NMFOnlineHALS(NMFBase):
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
        max_pass: int,
        chunk_size: int,
        w_max_iter: int,
        h_max_iter: int,
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

        self._max_pass = max_pass
        self._chunk_size = chunk_size
        self._w_max_iter = w_max_iter
        self._h_max_iter = h_max_iter
        self._zero = torch.tensor(0.0, dtype=self._tensor_dtype, device=self._device_type)


    def _h_err(self, h, hth, WWT, xWT):
        # Forbenious-norm^2 in trace format (No X)
        res = torch.trace(WWT @ hth) / 2.0 - torch.trace(h.T @ xWT)
        # Add regularization terms if needed
        if self._l1_reg_H > 0.0:
            res += self._l1_reg_H * h.norm(p=1)
        if self._l2_reg_H > 0.0:
            res += self._l2_reg_H * torch.trace(hth) / 2.0
        return res


    def _W_err(self, A, B, l1_reg_W, l2_reg_W, WWT):
        res = torch.trace(WWT @ A) / 2.0 - torch.trace(B @ self.W.T)
        # Add regularization terms if needed
        if l1_reg_W > 0.0:
            res += l1_reg_W * self.W.norm(p=1)
        if l2_reg_W > 0.0:
            res += l2_reg_W * torch.trace(WWT) / 2.0
        return res


    def _update_one_pass(self, l1_reg_W, l2_reg_W):
        indices = torch.randperm(self.X.shape[0], device=self._device_type)
        A = torch.zeros((self.k, self.k), dtype=self._tensor_dtype, device=self._device_type)
        B = torch.zeros((self.k, self.X.shape[1]), dtype=self._tensor_dtype, device=self._device_type)

        i = 0
        num_processed = 0
        WWT = self.W @ self.W.T
        while i < indices.shape[0]:
            idx = indices[i:(i+self._chunk_size)]
            cur_chunksize = idx.shape[0]
            x = self.X[idx, :]
            h = self.H[idx, :]

            # Online update H.
            hth = h.T @ h
            xWT = x @ self.W.T

            cur_h_err = self._h_err(h, hth, WWT, xWT)

            for j in range(self._h_max_iter):
                prev_h_err = cur_h_err

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
                    h[:, k] = hvec

                hth = h.T @ h
                cur_h_err = self._h_err(h, hth, WWT, xWT)

                if self._is_converged(prev_h_err, cur_h_err, prev_h_err):
                    break

            self.H[idx, :] = h

            # Update sufficient statistics A and B.
            num_after = num_processed + cur_chunksize

            A *= num_processed
            A += hth
            A /= num_after

            B *= num_processed
            B += h.T @ x
            B /= num_after

            num_processed = num_after

            # Online update W.
            cur_W_err = self._W_err(A, B, l1_reg_W, l2_reg_W, WWT)

            for j in range(self._w_max_iter):
                prev_W_err = cur_W_err

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
                    self.W[k, :] = w_new

                WWT = self.W @ self.W.T
                cur_W_err = self._W_err(A, B, l1_reg_W, l2_reg_W, WWT)

                if self._is_converged(prev_W_err, cur_W_err, prev_W_err):
                    break

            i += self._chunk_size

        return WWT


    def _update_H(self, WWT):
        i = 0
        sum_h_err = 0.0
        while i < self.H.shape[0]:
            x = self.X[i:(i+self._chunk_size), :]
            h = self.H[i:(i+self._chunk_size), :]

            hth = h.T @ h
            xWT = x @ self.W.T

            cur_h_err = self._h_err(h, hth, WWT, xWT)

            for j in range(self._h_max_iter):
                prev_h_err = cur_h_err

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
                    h[:, k] = hvec

                hth = h.T @ h
                cur_h_err = self._h_err(h, hth, WWT, xWT)

                if self._is_converged(prev_h_err, cur_h_err, prev_h_err):
                    break

            sum_h_err += cur_h_err
            i += self._chunk_size

        return sum_h_err


    @torch.no_grad()
    def fit(self, X):
        super().fit(X)
        assert self._beta==2, "Cannot perform online update when beta not equal to 2!"

        # Online update.
        self._chunk_size = min(self.X.shape[0], self._chunk_size)

        l1_reg_W = self._l1_reg_W / self.X.shape[0]
        l2_reg_W = self._l2_reg_W / self.X.shape[0]

        for i in range(self._max_pass):
            WWT = self._update_one_pass(l1_reg_W, l2_reg_W)
            H_err = self._update_H(WWT) # Update H again at the end of each pass.

            self._cur_err = torch.sqrt(2 * (H_err + self._X_SS_half + self._get_regularization_loss(self.W, self._l1_reg_W, self._l2_reg_W)))
            if self._is_converged(self._prev_err, self._cur_err, self._init_err):
                self.num_iters = i + 1
                print(f"    Converged after {self.num_iters} pass(es).")
                return

            self._prev_err = self._cur_err

        self.num_iters = self._max_pass
        print(f"    Not converged after {self._max_pass} pass(es).")


    def fit_transform(self, X):
        self.fit(X)
        return self.H
