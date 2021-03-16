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

        if self._init_method == 'nndsvdar':
            for k in range(self._n_batches):
                U, S, D = torch.svd_lowrank(self.X[k], q=self._n_components)

                H= torch.zeros_like(U, dtype=self._tensor_dtype, device=self._device_type)
                V = torch.zeros_like(D.T, dtype=self._tensor_dtype, device=self._device_type)
                H[:, 0] = S[0].sqrt() * U[:, 0]
                V[0, :] = S[0].sqrt() * D[:, 0]

                for j in range(2, self._n_components):
                    x, y = U[:, j], D[:, j]
                    x_p, y_p = x.maximum(torch.zeros_like(x, device=self._device_type)), y.maximum(torch.zeros_like(y, device=self._device_type))
                    x_n, y_n = x.minimum(torch.zeros_like(x, device=self._device_type)).abs(), y.minimum(torch.zeros_like(y, device=self._device_type)).abs()
                    x_p_nrm, y_p_nrm = x_p.norm(p=2), y_p.norm(p=2)
                    x_n_nrm, y_n_nrm = x_n.norm(p=2), y_n.norm(p=2)
                    m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

                    if m_p > m_n:
                        u, v, sigma = x_p / x_p_nrm, y_p / y_p_nrm, m_p
                    else:
                        u, v, sigma = x_n / x_n_nrm, y_n / y_n_nrm, m_n

                    factor = (S[j] * sigma).sqrt()
                    H[:, j] = factor * u
                    V[j, :] = factor * v

                H[H < eps] = 0
                V[V < eps] = 0

                if self._init_method == 'nndsvdar':
                    avg = self.X[k].mean()
                    H[H == 0] = avg / 100 * torch.rand(H[H==0].shape, dtype=self._tensor_dtype, device=self._device_type)
                    V[V == 0] = avg / 100 * torch.rand(V[V==0].shape, dtype=self._tensor_dtype, device=self._device_type)

                W += V / 2
                self.H.append(H)
                self.V.append(V)

            self.W = W / self._n_batches

        elif self._init_method == 'random':
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
        else:
            raise ValueError(f"Invalid init parameter. Got {self._init_method}, but require one of ('nndsvdar', 'random').")

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

    def _loss(self):
        res = 0.0
        for k in range(self._n_batches):
            res += torch.trace(self._HTH[k] @ self._WVWVT[k]) - 2 * torch.trace(self.H[k].T @ self._XWVT[k])
            if self._lambda > 0:
                res += self._lambda * torch.trace(self._VVT[k] @ self._HTH[k])
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

        self.X = mats
        # Cache Sum of squares of Xs.
        self._SSX = 0.0
        for k in range(self._n_batches):
            self._SSX += self.X[k].norm(p=2)**2

        self._initialize_W_H_V()
