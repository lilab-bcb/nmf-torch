import torch

from ._inmf_base import INMFBase
from typing import List, Union

class INMFBatchHALSWrong(INMFBase):
    def __init__(
        self,
        n_components: int,
        lam: float = 5.,
        init: str = 'random',
        tol: float = 1e-4,
        random_state: int = 0,
        fp_precision: Union[str, torch.dtype] = 'float',
        device_type: str = 'cpu',
        max_iter: int = 200,
    ):
        super().__init__(
            n_components=n_components,
            lam=lam,
            init=init,
            tol=tol,
            random_state=random_state,
            fp_precision=fp_precision,
            device_type=device_type,
        )

        self._max_iter = max_iter
        self._zero = torch.tensor(0.0, dtype=self._tensor_dtype, device=self._device_type)


    def _update_H_V_W(self):
        W_numer = torch.zeros_like(self.W)
        W_denom = torch.zeros_like(self._HTH[0])
        # Update Hs and Vs and calculate terms for updating W
        for k in range(self._n_batches):
            # Update H[k]
            for l in range(self._n_components):
                numer = self._XWVT[k][:, l] - self.H[k] @ self._WVWVT[k][:, l]
                if self._lambda > 0.0:
                    denom = self._WVWVT[k][l, l] + self._lambda * self._VVT[k][l, l]
                    h_new = self.H[k][:, l] * (self._WVWVT[k][l, l] / denom) + numer / denom
                else:
                    h_new = self.H[k][:, l] + numer / self._WVWVT[k][l, l]
                if torch.isnan(h_new).sum() > 0:
                    h_new[:] = 0.0 # divide zero error: set h_new to 0
                else:
                    h_new = h_new.maximum(self._zero)
                self.H[k][:, l] = h_new
            # Cache HTH
            self._HTH[k] = self.H[k].T @ self.H[k]

            # Update V[k]
            HTX = self.H[k].T @ self.X[k]
            for l in range(self._n_components):
                numer = HTX[l, :] - self._HTH[k][l, :] @ (self.W + self.V[k])
                denom = 1.0 + self._lambda
                v_new = self.V[k][l, :] * (1.0 / denom) + numer / (denom * self._HTH[k][l, l])
                if torch.isnan(v_new).sum() > 0:
                    v_new[:] = 0.0 # divide zero error: set v_new to 0
                else:
                    v_new = v_new.maximum(self._zero)
                self.V[k][l, :] = v_new
            # Cache VVT
            if self._lambda > 0.0:
                self._VVT[k] = self.V[k] @ self.V[k].T

            # Update W numer and denomer
            W_numer += (HTX - self._HTH[k] @ self.V[k])
            W_denom += self._HTH[k]

        # Update W
        for l in range(self._n_components):
            w_new = self.W[l, :] + (W_numer[l, :] - W_denom[l, :] @ self.W) / W_denom[l, l]
            if torch.isnan(w_new).sum() > 0:
                w_new[:] = 0.0 # divide zero error: set w_new to 0
            else:
                w_new = w_new.maximum(self._zero)
            self.W[l, :] = w_new
        # Cache WVWVT and XWVT
        for k in range(self._n_batches):
            WV = self.W + self.V[k]
            self._WVWVT[k] = WV @ WV.T
            self._XWVT[k] = self.X[k] @ WV.T


    def fit(
        self,
        mats: List[torch.tensor],
    ):
        super().fit(mats)

        # Batch update
        for i in range(self._max_iter):
            self._update_H_V_W()

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


    def fit_transform(
        self,
        mats: List[torch.tensor],
    ):
        self.fit(mats)
        return self.W
