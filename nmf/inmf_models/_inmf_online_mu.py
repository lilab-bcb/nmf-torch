import torch

from ._inmf_online_base import INMFOnlineBase
from typing import List, Union


class INMFOnlineMU(INMFOnlineBase):
    def __init__(
        self,
        n_components: int,
        lam: float,
        init: str,
        tol: float,
        n_jobs: int,
        random_state: int,
        fp_precision: Union[str, torch.dtype],
        device_type: str,
        max_pass: int,
        chunk_size: int,
        chunk_max_iter: int,
        h_tol: float,
        v_tol: float,
        w_tol: float,
    ):
        super().__init__(
            n_components=n_components,
            lam=lam,
            init=init,
            tol=tol,
            n_jobs=n_jobs,
            random_state=random_state,
            fp_precision=fp_precision,
            device_type=device_type,
            max_pass=max_pass,
            chunk_size=chunk_size,
        )

        self._chunk_max_iter = chunk_max_iter
        self._h_tol = h_tol
        self._v_tol = v_tol
        self._w_tol = w_tol


    def _update_matrix(self, mat, numer, denom):
        rates = numer / denom
        rates[denom < self._epsilon] = 0.0
        cur_max = (torch.abs(1.0 - rates) * mat).max()
        mat *= rates
        return cur_max


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

                h_factor_numer = xWVT
                for j in range(self._chunk_max_iter):
                    h_factor_denom = h @ (WVWVT + self._lambda * VVT) if self._lambda > 0.0 else h @ WVWVT
                    cur_max = self._update_matrix(h, h_factor_numer, h_factor_denom)
                    if j + 1 < self._chunk_max_iter and cur_max / h.mean() < self._h_tol:
                        break
                # print(f"Batch {k} Block {i} update H iterates {j+1} iterations.")
                self.H[k][idx, :] = h

                # Update sufficient statistics for batch k
                hth = h.T @ h
                A += hth
                htx = h.T @ x
                B += htx

                # Update V
                V_factor_numer = B
                for j in range(self._chunk_max_iter):
                    V_factor_denom = A @ (WV + self._lambda * self.V[k])
                    cur_max = self._update_matrix(self.V[k], V_factor_numer, V_factor_denom)
                    WV = self.W + self.V[k]
                    if j + 1 < self._chunk_max_iter and cur_max / self.V[k].mean() < self._v_tol:
                        break
                # print(f"Batch {k} Block {i} update V iterates {j+1} iterations.")

                # Update sufficient statistics for all batches
                C += hth
                D += htx
                E_new = E + A @ self.V[k]

                # Update W
                W_factor_numer = D
                for j in range(self._chunk_max_iter):
                    W_factor_denom = C @ self.W + E_new
                    cur_max = self._update_matrix(self.W, W_factor_numer, W_factor_denom)
                    if j + 1 < self._chunk_max_iter and cur_max / self.W.mean() < self._w_tol:
                        break
                # print(f"Batch {k} Block {i} update W iterates {j+1} iterations.")
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

                h_factor_numer = xWVT
                for j in range(self._chunk_max_iter):
                    h_factor_denom = h @ (WVWVT + self._lambda * VVT) if self._lambda > 0.0 else h @ WVWVT
                    cur_max = self._update_matrix(h, h_factor_numer, h_factor_denom)
                    if j + 1 < self._chunk_max_iter and cur_max / h.mean() < self._h_tol:
                        break
                # print(f"Batch {k} Block {i} update H iterates {j+1} iterations.")
                self.H[k][idx, :] = h

                # Update sufficient statistics for batch k
                hth = h.T @ h
                A += hth
                htx = h.T @ x
                B += htx

                # Update V
                V_factor_numer = B
                for j in range(self._chunk_max_iter):
                    V_factor_denom = A @ (WV + self._lambda * self.V[k])
                    cur_max = self._update_matrix(self.V[k], V_factor_numer, V_factor_denom)
                    WV = self.W + self.V[k]
                    if j + 1 < self._chunk_max_iter and cur_max / self.V[k].mean() < self._v_tol:
                        break
                # print(f"Batch {k} Block {i} update V iterates {j+1} iterations.")
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
                h_factor_numer = xWVT
                for j in range(self._chunk_max_iter):
                    h_factor_denom = h @ (WVWVT + self._lambda * VVT) if self._lambda > 0.0 else h @ WVWVT
                    cur_max = self._update_matrix(h, h_factor_numer, h_factor_denom)
                    if j + 1 < self._chunk_max_iter and cur_max / h.mean() < self._h_tol:
                            break
                # print(f"Batch {k} Block {i} update H iterates {j+1} iterations.")

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
