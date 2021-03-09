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
        avg = torch.sqrt(self.X[0].mean() / self._n_components)
        self.W = torch.abs(avg * torch.randn((self._n_components, self._n_features), dtype=self._tensor_dtype, device=self._device_type))
        W_t = self.W.T
        self.H = []
        self.V = []
        init_err = self._SSX.clone()
        self._H_t = []
        self._HTH = []
        self._WVWVT = []
        self._XWVT = []
        for k in range(self._n_batches):
            avg = torch.sqrt(self.X[k].mean() / self._n_components)
            H = torch.abs(avg * torch.randn((self.X[k].shape[0], self._n_components), dtype=self._tensor_dtype, device=self._device_type))
            V = torch.abs(avg * torch.randn((self._n_components, self._n_features), dtype=self._tensor_dtype, device=self._device_type))
            self.H.append(H)
            self.V.append(V)

            # Cache for batch update
            WV = self.W + V
            WV_t = WV.T
            self._H_t.append(H.T)
            self._HTH.append(self._H_t[k] @ H)
            self._WVWVT.append(WV @ WV_t)
            self._XWVT.append(self.X[k] @ WV_t)

            # Calculate loss of this batch
            init_err += self._loss(batch_id=k)
        self._init_err = torch.sqrt(init_err)
        self._prev_err = self._init_err
        self._cur_err = self._init_err


    def _loss(self, batch_id=None):
        if batch_id is None:
            res = self._SSX.clone()
            for k in range(self._n_batches):
                res += torch.trace(self._HTH[k] @ self._WVWVT[k]) - 2 * torch.trace(self._H_t[k] @ self._XWVT[k])
                if self._lambda > 0:
                    res += torch.sum((self.H[k]@self.V[k])**2)
            return torch.sqrt(res)
        else:
            assert batch_id >= 0 and batch_id < self._n_batches, "Batch ID is out of bound!"
            return torch.trace(self._HTH[batch_id] @ self._WVWVT[batch_id]) - 2 * torch.trace(self._H_t[batch_id] @ self._XWVT[batch_id])


    @property
    def reconstruction_err(self):
        return self._cur_err


    def _is_converged(self, prev_err, cur_err, init_err):
        return torch.abs((prev_err - cur_err) / init_err) < self._tol


    def _update_matrix(self, mat, numer, denom):
        mat *= (numer / denom)
        mat[denom < self._epsilon] = 0.0


    def _update_W(self):
        W_numer = torch.zeros_like(self.W)
        W_denom = torch.zeros_like(self.W)
        for k in range(self._n_batches):
            W_numer += self._H_t[k] @ self.X[k]
            W_denom += self._HTH[k] @ (self.W + self._lambda * self.V[k])
        self._update_matrix(self.W, W_numer, W_denom)


    def _update_H_V(self):
        for k in range(self._n_batches):
            # Update V[k]
            V_numer = self._H_t[k] @ self.X[k]
            V_denom = self._HTH[k] @ (self.W + self.V[k])
            if self._lambda > 0:
                V_denom += self._lambda * (self._HTH[k] @ self.V[k])
            self._update_matrix(self.V[k], V_numer, V_denom)

            # Cache WVWVT and XWVT
            WV = self.W + self.V[k]
            WV_t = WV.T
            self._WVWVT[k] = WV @ WV_t
            self._XWVT[k] = self.X[k] @ WV_t

            # Update H[k]
            H_numer = self._XWVT[k].clone()
            H_denom = self.H[k] @ self._WVWVT[k]
            if self._lambda > 0:
                H_denom += self._lambda * (self.H[k] @ (self.V[k] @ self.V[k].T))
            self._update_matrix(self.H[k], H_numer, H_denom)

            # Cache H_t and HTH
            self._H_t[k] = self.H[k].T
            self._HTH[k] = self._H_t[k] @ self.H[k]

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

            self._update_W()
            self._update_H_V()

            if i == self._max_iter - 1:
                self.num_iters = self._max_iter
                print(f"    Not converged after {self.num_iters} iteration(s).")


    def fit_transform(
        self,
        mats: List[torch.tensor],
    ):
        self.fit(mats)
        return self.W
