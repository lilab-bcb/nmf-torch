import torch

from typing import List, Union

class INMFBatch:
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
        self._max_iter = max_iter


    def _initialize_W_H_V(self):
        # Random initialization
        W = torch.zeros((self._n_components, self._n_features), dtype=self._tensor_dtype, device=self._device_type)
        for k in range(self._n_batches):
            avg = torch.sqrt(self.X[k].mean() / self._n_components)
            H = torch.abs(avg * torch.randn((self.X[k].shape[0], self._n_components), dtype=self._tensor_dtype, device=self._device_type))
            V = torch.abs(0.5 * avg * torch.randn((self._n_components, self._n_features), dtype=self._tensor_dtype, device=self._device_type))
            self.H.append(H)
            self.V.append(V)
            W += torch.abs(0.5 * avg * torch.randn((self._n_components, self._n_features), dtype=self._tensor_dtype, device=self._device_type))
        W /= self._n_batches

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
            res += torch.trace(self._HTH[k] @ self._WVWVT[k]) - 2 * torch.trace(self._H_t[k] @ self._XWVT[k])
            if self._lambda > 0:
                res += torch.trace(self._VVT[k] @ self._HTH[k])
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
        torch.manual_seed(self._random_state)
        self._n_batches = len(mats)
        if self._n_batches <= 1:
            print("    No need to integrate!")
            return

        self._n_features = mats[0].shape[1]
        for i in range(self._n_batches):
            if mats[i].shape[1] != self._n_features:
                raise ValueError("Number of features must be the same across samples!")

        self.X = mats
        # Cache Sum of squares of Xs.
        self._SSX = 0.0
        for k in range(self._n_batches):
            self._SSX += torch.sum(self.X[k]**2)

        self._initialize_W_H_V()

        # Batch update
        for i in range(self._max_iter):
            if (i + 1) % 10 == 0:
                self._cur_err = self._loss()
                if self._is_converged(self._prev_err, self._cur_err, self._init_err):
                    self.num_iters = i + 1
                    print(f"    Converged after {self.num_iters} iteration(s).")
                    break
                else:
                    self._prev_err = self._cur_err

            self._update_H_V_W()

            if i == self._max_iter - 1:
                self.num_iters = self._max_iter
                print(f"    Not converged after {self.num_iters} iteration(s).")


    def fit_transform(
        self,
        mats: List[torch.tensor],
    ):
        self.fit(mats)
        return self.W
