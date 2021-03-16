import torch

from ._inmf_base import INMFBase
from typing import List, Union

class INMFBatch(INMFBase):
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


    def _update_H_V_W(self):
        W_numer = torch.zeros_like(self.W)
        W_denom = torch.zeros_like(self.W)
        # Update Hs and Vs and calculate partials sum for updating W
        for k in range(self._n_batches):
            # Update H[k]
            H_numer = self._XWVT[k]
            H_denom = self.H[k] @ self._WVWVT[k]
            if self._lambda > 0:
                H_denom += self._lambda * (self.H[k] @ self._VVT[k])
            self._update_matrix(self.H[k], H_numer, H_denom)
            # Cache HTH
            self._HTH[k] = self.H[k].T @ self.H[k]

            # Update V[k]
            V_numer = self.H[k].T @ self.X[k]
            V_denom = self._HTH[k] @ (self.W + self.V[k])
            if self._lambda > 0:
                V_denom += self._lambda * (self._HTH[k] @ self.V[k])
            self._update_matrix(self.V[k], V_numer, V_denom)

            # Update W numer and denomer
            W_numer += V_numer
            W_denom += self._HTH[k] @ (self.W + self.V[k])
        # Update W
        self._update_matrix(self.W, W_numer, W_denom)
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
