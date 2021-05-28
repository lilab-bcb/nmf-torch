import torch

from typing import List, Union

class INMFBase:
    def __init__(
        self,
        n_components: int,
        lam: float = 5.,
        init: str = 'random',
        tol: float = 1e-4,
        random_state: int = 0,
        fp_precision: Union[str, torch.dtype] = 'float',
        device_type: str = 'cpu',
    ):
        self._n_components = n_components
        self._init_method = init
        self._lambda = lam
        self._tol = tol
        self._random_state = random_state
        self._epsilon = 1e-20

        if fp_precision == 'float':
            self._tensor_dtype = torch.float
        elif fp_precision == 'double':
            self._tensor_dtype = torch.double
        else:
            self._tensor_dtype = fp_precision

        self._device_type = device_type


    def _initialize_W_H_V(self, eps=1e-6):
        W = torch.zeros((self._n_components, self._n_features), dtype=self._tensor_dtype, device=self._device_type)
        self.H = []
        self.V = []

        # Random initialization
        for k in range(self._n_batches):
            avg = torch.sqrt(self.X[k].mean() / self._n_components)
            H = torch.abs(avg * torch.randn((self.X[k].shape[0], self._n_components), dtype=self._tensor_dtype, device=self._device_type))
            V = torch.abs(0.5 * avg * torch.randn((self._n_components, self._n_features), dtype=self._tensor_dtype, device=self._device_type))
            self.H.append(H)
            self.V.append(V)
            W += torch.abs(0.5 * avg * torch.randn((self._n_components, self._n_features), dtype=self._tensor_dtype, device=self._device_type))
        W /= self._n_batches
        self.W = W

        self._HTH = []
        self._WVWVT = []
        self._XWVT = []
        self._VVT = []
        for k in range(self._n_batches):
            # Cache for batch update
            WV = self.W + V
            self._HTH.append(self.H[k].T @ self.H[k])
            WV = self.W + self.V[k]
            self._WVWVT.append(WV @ WV.T)
            self._XWVT.append(self.X[k] @ WV.T)
            if self._lambda > 0.0:
                self._VVT.append(self.V[k] @ self.V[k].T)
        self._init_err = self._loss()
        self._prev_err = self._init_err
        self._cur_err = self._init_err

    def _trace(self, A, B):
        # return trace(A.T @ B) or trace(A @ B.T)
        return torch.dot(A.ravel(), B.ravel())

    def _loss(self):
        res = 0.0
        for k in range(self._n_batches):
            res += self._trace(self._HTH[k], self._WVWVT[k]) - 2.0 * self._trace(self.H[k], self._XWVT[k])
            if self._lambda > 0.0:
                res += self._lambda * self._trace(self._VVT[k], self._HTH[k])
        res += self._SSX
        return torch.sqrt(res)

    @property
    def reconstruction_err(self):
        return self._cur_err

    def _is_converged(self, prev_err, cur_err, init_err):
        return torch.abs((prev_err - cur_err) / init_err) < self._tol

    def _update_matrix(self, mat, numer, denom):
        mat *= (numer / denom)
        mat[denom < self._epsilon] = 0.0

    def fit(
        self,
        mats: List[torch.tensor],
    ):
        torch.manual_seed(self._random_state)
        self._n_batches = len(mats)
        if self._n_batches <= 1:
            print("    No need to integrate!")
            return

        self._n_features = mats[0].shape[1]
        for i in range(self._n_batches):
            if mats[i].shape[1] != self._n_features:
                raise ValueError(f"Number of features must be the same across samples, while Sample {i} is not!")
            if torch.any(mats[i] < 0):
                raise ValueError(f"Input matrix {i} is not non-negative. NMF cannot be applied.")

        if self._device_type == 'cpu':
            self.X = mats
        else:
            self.X = []
            for X in mats:
                self.X.append(X.cuda())

        # Cache Sum of squares of Xs.
        self._SSX = 0.0
        for k in range(self._n_batches):
            self._SSX += self.X[k].norm(p=2)**2

        self._initialize_W_H_V()
