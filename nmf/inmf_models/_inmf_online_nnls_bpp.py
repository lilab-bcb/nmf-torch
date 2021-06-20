import torch

from ._inmf_online_base import INMFOnlineBase
from ..utils import nnls_bpp
from typing import List


class INMFOnlineNnlsBpp(INMFOnlineBase):
    def _update_one_pass(self):
        """
            A = sum hth; B = sum htx; for each batch
            C = sum of hth; D = sum of htx; E = sum of AV; for all batches
        """
        A = torch.zeros((self._n_components, self._n_components), dtype=self._tensor_dtype, device=self._device_type)
        B = torch.zeros((self._n_components, self._n_features), dtype=self._tensor_dtype, device=self._device_type)
        C = torch.zeros((self._n_components, self._n_components), dtype=self._tensor_dtype, device=self._device_type)
        D = torch.zeros((self._n_components, self._n_features), dtype=self._tensor_dtype, device=self._device_type)
        E = torch.zeros((self._n_components, self._n_features), dtype=self._tensor_dtype, device=self._device_type)

        batch_indices = torch.randperm(self._n_batches, device=self._device_type)
        for k in batch_indices:
            indices = torch.randperm(self.X[k].shape[0], device=self._device_type)

            # Block-wise update
            i = 0
            A.fill_(0.0)
            B.fill_(0.0)
            while i < indices.shape[0]:
                idx = indices[i:(i+self._chunk_size)]
                x = self.X[k][idx, :]
                h = self.H[k][idx, :]

                # Update H
                WV = self.W + self.V[k]
                WVWVT = WV @ WV.T
                VVT = self.V[k] @ self.V[k].T if self._lambda > 0.0 else None
                xWVT = x @ WV.T

                if self._lambda > 0.0:
                    n_iter = nnls_bpp(WVWVT + self._lambda * VVT, xWVT.T, h.T, self._device_type)
                else:
                    n_iter = nnls_bpp(WVWVT, xWVT.T, h.T, self._device_type)
                # print(f"Batch {k} Block {i} H n_iter={n_iter}.")
                self.H[k][idx, :] = h

                # Update sufficient statistics for batch k
                hth = h.T @ h
                A += hth
                htx = h.T @ x
                B += htx

                # Update V
                n_iter = nnls_bpp(A * (1.0 + self._lambda), B - A @ self.W, self.V[k], self._device_type)
                # print(f"Batch {k} Block {i} V n_iter={n_iter}.")

                # Update sufficient statistics for all batches
                C += hth
                D += htx
                E_new = E + A @ self.V[k]

                # Update W
                n_iter = nnls_bpp(C, D - E_new, self.W, self._device_type)
                # print(f"Batch {k} Block {i} W n_iter={n_iter}.")

                i += self._chunk_size
            E = E_new


    def _update_H_V(self):
        """
            Fix W, only update V and H
            A = sum hth; B = sum htx; for each batch
        """
        A = torch.zeros((self._n_components, self._n_components), dtype=self._tensor_dtype, device=self._device_type)
        B = torch.zeros((self._n_components, self._n_features), dtype=self._tensor_dtype, device=self._device_type)

        for k in range(self._n_batches):
            indices = torch.randperm(self.X[k].shape[0], device=self._device_type)

            # Block-wise update
            i = 0
            A.fill_(0.0)
            B.fill_(0.0)
            while i < indices.shape[0]:
                idx = indices[i:(i+self._chunk_size)]
                x = self.X[k][idx, :]
                h = self.H[k][idx, :]

                # Update H
                WV = self.W + self.V[k]
                WVWVT = WV @ WV.T
                VVT = self.V[k] @ self.V[k].T if self._lambda > 0.0 else None
                xWVT = x @ WV.T

                if self._lambda > 0.0:
                    n_iter = nnls_bpp(WVWVT + self._lambda * VVT, xWVT.T, h.T, self._device_type)
                else:
                    n_iter = nnls_bpp(WVWVT, xWVT.T, h.T, self._device_type)
                # print(f"Batch {k} Block {i} H n_iter={n_iter}.")
                self.H[k][idx, :] = h

                # Update sufficient statistics for batch k
                hth = h.T @ h
                A += hth
                htx = h.T @ x
                B += htx

                # Update V
                n_iter = nnls_bpp(A * (1.0 + self._lambda), B - A @ self.W, self.V[k], self._device_type)
                # print(f"Batch {k} Block {i} V n_iter={n_iter}.")

                i += self._chunk_size


    def _update_H(self):
        """ Fix W and V, update H """
        sum_h_err = torch.tensor(0.0, dtype=torch.double, device=self._device_type) # make sure sum_h_err is double to avoid summation errors
        for k in range(self._n_batches):
            WV = self.W + self.V[k]
            WVWVT = WV @ WV.T
            VVT = self.V[k] @ self.V[k].T if self._lambda > 0.0 else None

            i = 0
            while i < self.H[k].shape[0]:
                x = self.X[k][i:(i+self._chunk_size), :]
                h = self.H[k][i:(i+self._chunk_size), :]

                # Update H
                xWVT = x @ WV.T
                if self._lambda > 0.0:
                    n_iter = nnls_bpp(WVWVT + self._lambda * VVT, xWVT.T, h.T, self._device_type)
                else:
                    n_iter = nnls_bpp(WVWVT, xWVT.T, h.T, self._device_type)
                # print(f"Batch {k} Block {i} H n_iter={n_iter}.")

                hth = h.T @ h
                sum_h_err += self._h_err(h, hth, WVWVT, xWVT, VVT)
                i += self._chunk_size

        return torch.sqrt(sum_h_err + self._SSX)


    def fit(
        self,
        mats: List[torch.tensor],
    ):
        super().fit(mats)

        self.num_iters = -1
        for i in range(self._max_pass):
            self._update_one_pass()
            self._cur_err = self._loss()
            print(f"pass {i+1}: loss={self._cur_err}.")

            if self._is_converged(self._prev_err, self._cur_err, self._init_err):
                self.num_iters = i + 1
                print(f"    Converged after {self.num_iters} pass(es).")
                break

            self._prev_err = self._cur_err

        if self.num_iters < 0:
            self.num_iters = self._max_pass
            print(f"    Not converged after {self._max_pass} pass(es).")

        self._update_H_V()
        self._cur_err = self._update_H()
        print(f"Final loss={self._cur_err}.")
