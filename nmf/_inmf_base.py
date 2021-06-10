import numpy
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


    def _initialize_W_H_V(self):
        ##W = torch.zeros((self._n_components, self._n_features), dtype=self._tensor_dtype, device=self._device_type)
        ##self.H = []
        ##self.V = []

        # Random initialization
        ##for k in range(self._n_batches):
        ##    avg = torch.sqrt(self.X[k].mean() / self._n_components)
        ##    H = torch.abs(avg * torch.randn((self.X[k].shape[0], self._n_components), dtype=self._tensor_dtype, device=self._device_type))
        ##    V = torch.abs(0.5 * avg * torch.randn((self._n_components, self._n_features), dtype=self._tensor_dtype, device=self._device_type))
        ##    self.H.append(H)
        ##    self.V.append(V)
        ##    W += torch.abs(0.5 * avg * torch.randn((self._n_components, self._n_features), dtype=self._tensor_dtype, device=self._device_type))
        ##W /= self._n_batches
        ##self.W = W
        W = 2.0 * torch.rand((self._n_components, self._n_features), dtype=self._tensor_dtype, device=self._device_type)
        self.H = []
        self.V = []
        for k in range(self._n_batches):
            H = 2.0 * torch.rand((self.X[k].shape[0], self._n_components), dtype=self._tensor_dtype, device=self._device_type)
            V = 2.0 * torch.rand((self._n_components, self._n_features), dtype=self._tensor_dtype, device=self._device_type)
            self.H.append(H)
            self.V.append(V)


    def _trace(self, A, B):
        # return trace(A.T @ B) or trace(A @ B.T)
        return torch.dot(A.ravel(), B.ravel())

    def _loss(self): # not defined here
        return None

    @property
    def reconstruction_err(self):
        return self._cur_err

    def _is_converged(self, prev_err, cur_err, init_err):
        return prev_err <= cur_err or torch.abs((prev_err - cur_err) / init_err) < self._tol


    def _cast_tensor(self, X):
        if not isinstance(X, torch.Tensor):
            if self._device_type == 'cpu' and ((self._device_type == torch.float32 and X.dtype == numpy.float32) or (self._device_type == torch.double and X.dtype == numpy.float64)):
                X = torch.from_numpy(X)
            else:    
                X = torch.tensor(X, dtype=self._tensor_dtype, device=self._device_type)
        else:
            if self._device_type != 'cpu' and (not X.is_cuda):
                X = X.to(device=self._device_type)
            if X.dtype != self._tensor_dtype:
                X = X.type(self._tensor_dtype)
        return X


    def fit(
        self,
        mats: Union[List[numpy.ndarray], List[torch.tensor]],
    ):
        torch.manual_seed(self._random_state)
        self._n_batches = len(mats)
        if self._n_batches <= 1:
            print(f"    Contains only {self._n_batches}, no need to integrate!")
            return

        self._n_features = mats[0].shape[1]
        for i in range(self._n_batches):
            mats[i] = self._cast_tensor(mats[i])
            if mats[i].shape[1] != self._n_features:
                raise ValueError(f"Number of features must be the same across samples, while Sample {i} is not!")
            if torch.any(mats[i] < 0):
                raise ValueError(f"Input matrix {i} is not non-negative. NMF cannot be applied.")
        self.X = mats

        # Cache Sum of squares of Xs.
        self._SSX = torch.tensor(0.0, dtype=torch.double, device=self._device_type) # make sure _SSX is double to avoid summation errors
        for k in range(self._n_batches):
            self._SSX += self.X[k].norm(p=2)**2

        self._initialize_W_H_V()


    def fit_transform(
        self,
        mats: List[torch.tensor],
    ):
        self.fit(mats)
        return self.W
