import torch

from ._inmf_base import INMFBase
from typing import List, Union


class INMFOnlineBase(INMFBase):
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
        )

        self._max_pass = max_pass
        self._chunk_size = chunk_size


    def _h_err(self, h, hth, WVWVT, xWVT, VVT):
        # Calculate L2 Loss (no sum of squares of X) for block h in trace format.
        res = self._trace(WVWVT + self._lambda * VVT, hth) if self._lambda > 0.0 else self._trace(WVWVT, hth)
        res -= 2.0 * self._trace(h, xWVT)
        return res


    def _loss(self):
        """ calculate loss online by passing through all data"""
        sum_h_err = torch.tensor(0.0, dtype=torch.double, device=self._device_type) # make sure sum_h_err is double to avoid summation errors
        for k in range(self._n_batches):
            WV = self.W + self.V[k]
            WVWVT = WV @ WV.T
            VVT = self.V[k] @ self.V[k].T if self._lambda > 0.0 else None

            i = 0
            while i < self.H[k].shape[0]:
                x = self.X[k][i:(i+self._chunk_size), :]
                h = self.H[k][i:(i+self._chunk_size), :]
                hth = h.T @ h
                xWVT = x @ WV.T
                sum_h_err += self._h_err(h, hth, WVWVT, xWVT, VVT)
                i += self._chunk_size

        return torch.sqrt(sum_h_err + self._SSX)


    def fit(
        self,
        mats: List[torch.tensor],
    ):
        super().fit(mats)

        self._init_err = self._loss()
        self._prev_err = self._init_err
