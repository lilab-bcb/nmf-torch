import torch

from ._inmf_base import INMFBase
from typing import List, Union


class INMFBatchBase(INMFBase):
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
        max_iter: int,
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

        self._max_iter = max_iter


    def _loss(self):
        res = torch.tensor(0.0, dtype=torch.double, device=self._device_type) # make sure res is double to avoid summation errors
        for k in range(self._n_batches):
            res += self._trace(self._HTH[k], self._WVWVT[k]) - 2.0 * self._trace(self.H[k], self._XWVT[k])
            if self._lambda > 0.0:
                res += self._lambda * self._trace(self._VVT[k], self._HTH[k])
        res += self._SSX
        return torch.sqrt(res)


    def fit(
        self,
        mats: List[torch.tensor],
    ):
        super().fit(mats)

        # Cache for batch update
        self._HTH = []
        self._WVWVT = []
        self._XWVT = []
        self._VVT = []
        for k in range(self._n_batches):
            self._HTH.append(self.H[k].T @ self.H[k])
            WV = self.W + self.V[k]
            self._WVWVT.append(WV @ WV.T)
            self._XWVT.append(self.X[k] @ WV.T)
            if self._lambda > 0.0:
                self._VVT.append(self.V[k] @ self.V[k].T)

        # Calculate init_err
        self._init_err = self._loss()
        self._prev_err = self._init_err
