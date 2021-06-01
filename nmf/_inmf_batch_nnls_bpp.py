import torch

from ._inmf_base import INMFBase
from nmf.cylib.nnls_bpp_utils import nnls_bpp

from typing import List, Union

class INMFBatchNnlsBpp(INMFBase):
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
        assert self._lambda > 0.0
        W_numer = torch.zeros_like(self.W)
        W_denom = torch.zeros_like(self._HTH[0])
        # Update Hs and Vs and calculate terms for updating W
        for k in range(self._n_batches):
            # Update H[k]
            n_iter = nnls_bpp(self._WVWVT[k] + self._lambda * self._VVT[k], self._XWVT[k].T, self.H[k].T, self._device_type)
            # Cache HTH
            self._HTH[k] = self.H[k].T @ self.H[k]

            # Update V[k]
            HTX = self.H[k].T @ self.X[k]
            n_iter = nnls_bpp(self._HTH[k] * (1.0 + self._lambda), HTX - self._HTH[k] @ self.W, self.V[k], self._device_type)
            # Cache VVT
            if self._lambda > 0.0:
                self._VVT[k] = self.V[k] @ self.V[k].T

            # Update W numer and denomer
            W_numer += (HTX - self._HTH[k] @ self.V[k])
            W_denom += self._HTH[k]

        # Update W
        n_iter = nnls_bpp(W_denom, W_numer, self.W, self._device_type)
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

            print(f" niter={i+1}, loss={self._loss()}.")
            if (i + 1) % 10 == 0:
                self._cur_err = self._loss()
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
