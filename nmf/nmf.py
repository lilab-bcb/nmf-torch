import numpy as np
import random
import torch

class NMF:
    def __init__(self, n_components, loss='l2', beta = 2, max_iter=200, tol=1e-4, random_state=0, fp_precision='float'):
        self.k = n_components

        if loss == 'l2':
            self._beta = 2
        if loss == 'kullback-leibler':
            self._beta = 1
        elif loss == 'itakura-saito':
            self._beta = 0
        else:
            self._beta = beta

        if fp_precision == 'float':
            self._tensor_dtype = torch.float
        elif fp_precision == 'double':
            self._tensor_dtype = torch.double
        else:
            self._tensor_dtype = fp_precision

        self._max_iter = max_iter
        self._tol = tol
        self._random_state = random_state
        self._deviance = []

    def _loss(self):
        if self._beta == 0:
            x_div_x_hat = self.X / self.X_hat
            diver_mat = x_div_x_hat - torch.log2(x_div_x_hat) - 1
        elif self._beta == 1:
            diver_mat = self.X * torch.log2(self.X / self.X_hat) - self.X + self.X_hat
        else:
            coef = 1 / (self._beta * (self._beta - 1))
            diver_mat = coef * (self.X.pow(self._beta) + (self._beta - 1) * self.X_hat.pow(self._beta) - self._beta * self.X * self.X_hat.pow(self._beta - 1))
        
        return torch.sum(diver_mat)

    def _update_loss(self):
        self._deviance.append(self._loss())
    
    def _is_converged(self):
        if len(self._deviance) > 1:
            return torch.abs((self._deviance[-1] - self._deviance[-2]) / self._deviance[0]) < self._tol
        else:
            return False

    def _initialize_W_H(self):
        random.seed(self._random_state)

        self.H = torch.rand((self.X.shape[0], self.k), dtype=self._tensor_dtype)
        self.W = torch.rand((self.k, self.X.shape[1]), dtype=self._tensor_dtype)
        self.X_hat = self.H @ self.W
        self._update_loss()

    @torch.no_grad()
    def fit(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self._tensor_dtype)
        elif X.dtype != self._tensor_dtype:
            X = X.type(self._tensor_dtype)
        assert torch.sum(X<0) == 0, "The input matrix is not non-negative. NMF cannot be applied."

        self.X = X
        self._initialize_W_H()

        for i in range(self._max_iter):
            if self._is_converged():
                print(f"    Reach convergence after {i+1} iteration(s).")
                break
            
            # Batch update on H.
            W_t = self.W.T
            H_factor_numer = (self.X * self.X_hat.pow(self._beta - 2)) @ W_t
            H_factor_denom = self.X_hat.pow(self._beta - 1) @ W_t
            self.H *= (H_factor_numer / H_factor_denom)

            # Batch update on W.
            H_t = self.H.T
            W_factor_numer = H_t @ (self.X * self.X_hat.pow(self._beta - 2))
            W_factor_denom = H_t @ self.X_hat.pow(self._beta - 1)
            self.W *= (W_factor_numer / W_factor_denom)

            self.X_hat = self.H @ self.W
            self._update_loss()

            if i == self._max_iter - 1:
                print(f"    Not converged after {self._max_iter} iteration(s).")

        return self.W
