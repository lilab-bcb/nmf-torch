import torch

from typing import Union
from ._nmf_online_base import NMFOnlineBase


class NMFOnlineMU(NMFOnlineBase):
    def _update_matrix(self, mat, numer, denom):
        rates = numer / denom
        rates[denom < self._epsilon] = 0.0
        cur_max = (torch.abs(1.0 - rates) * mat).max()
        mat *= rates
        return cur_max


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

            if self._l1_reg_H > 0.0:
                h_factor_numer = xWT - self._l1_reg_H
                h_factor_numer[h_factor_numer < 0.0] = 0.0
            else:
                h_factor_numer = xWT

            for j in range(self._chunk_max_iter):
                h_factor_denom = h @ WWT
                if self._l2_reg_H:
                    h_factor_denom += self._l2_reg_H * h
                cur_max = self._update_matrix(h, h_factor_numer, h_factor_denom)
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
            if l1_reg_W > 0.0:
                W_factor_numer = B - l1_reg_W
                W_factor_numer[W_factor_numer < 0.0] = 0.0
            else:
                W_factor_numer = B

            for j in range(self._chunk_max_iter):
                W_factor_denom = A @ self.W
                if l2_reg_W > 0.0:
                    W_factor_denom += l2_reg_W * self.W
                cur_max = self._update_matrix(self.W, W_factor_numer, W_factor_denom)
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

            if self._l1_reg_H > 0.0:
                h_factor_numer = xWT - self._l1_reg_H
                h_factor_numer[h_factor_numer < 0.0] = 0.0
            else:
                h_factor_numer = xWT

            for j in range(self._chunk_max_iter):
                h_factor_denom = h @ WWT
                if self._l2_reg_H:
                    h_factor_denom += self._l2_reg_H * h
                cur_max = self._update_matrix(h, h_factor_numer, h_factor_denom)
                if j + 1 < self._chunk_max_iter and cur_max / h.mean() < self._h_tol:
                    break

            # print(f"Block {i} update H iterates {j+1} iterations.")

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