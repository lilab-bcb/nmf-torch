import numpy as np
import random
import torch

class NMF:
    def __init__(self, n_components, init='nndsvd', loss='l2', beta = 2, max_iter=200, tol=1e-4, random_state=0, fp_precision='float', use_gpu=False):
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

        self._device_type = 'cpu'
        if use_gpu:
            if torch.cuda.is_available():
                self._device_type = 'cuda'
                print("Use GPU mode.")
            else:
                print("CUDA is not available on your machine. Use CPU mode instead.")

        self._init_method = init
        self._max_iter = max_iter
        self._tol = tol
        self._random_state = random_state
        self._deviance = []

    @staticmethod
    def _loss(X, Y, beta):
        if beta == 0:
            x_div_y = X / Y
            diver_mat = x_div_y - x_div_y.log2() - 1
        elif beta == 1:
            diver_mat = X * torch.log2(X / Y) - X + Y
        else:
            coef = 1 / (beta * (beta - 1))
            diver_mat = coef * (X.pow(beta) + (beta - 1) * Y.pow(beta) - beta * X * Y.pow(beta - 1))
        
        return diver_mat.sum()

    def _update_loss(self, X_hat):
        self._deviance.append(self._loss(self.X, X_hat, self._beta))
    
    def _is_converged(self):
        if len(self._deviance) > 1:
            return torch.abs((self._deviance[-1] - self._deviance[-2]) / self._deviance[0]) < self._tol
        else:
            return False

    def _get_X_hat(self):
        return self.H @ self.W

    def _initialize_W_H(self, eps=1e-6):
        random.seed(self._random_state)

        if self._init_method == 'nndsvd':
            U, S, V = torch.svd_lowrank(self.X, q=self.k)
            H, W = torch.zeros_like(U), torch.zeros_like(V.T)
            H[:, 0] = S[0].square() * U[:, 0]
            W[0, :] = S[0].square() * V[:, 0]

            for j in range(2, self.k):
                x, y = U[:, j], V[:, j]
                x_p, y_p = x.maximum(torch.zeros_like(x)), y.maximum(torch.zeros_like(y))
                x_n, y_n = x.minimum(torch.zeros_like(x)).abs(), y.minimum(torch.zeros_like(y)).abs()
                x_p_nrm, y_p_nrm = x_p.norm(p=2), y_p.norm(p=2)
                x_n_nrm, y_n_nrm = x_n.norm(p=2), y_n.norm(p=2)
                m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

                if m_p > m_n:
                    u, v, sigma = x_p / x_p_nrm, y_p / y_p_nrm, m_p
                else:
                    u, v, sigma = x_n / x_n_nrm, y_n / y_n_nrm, m_n
                
                factor = (S[j] * sigma).sqrt()
                H[:, j] = factor * u
                W[j, :] = factor * v

            H[H < eps] = 0
            W[W < eps] = 0
        else:
            H = torch.rand((self.X.shape[0], self.k), dtype=self._tensor_dtype, device=self._device_type)
            W = torch.rand((self.k, self.X.shape[1]), dtype=self._tensor_dtype, device=self._device_type)
        
        self.H = H
        self.W = W
        self._update_loss(self._get_X_hat())

    @torch.no_grad()
    def fit(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self._tensor_dtype, device=self._device_type)
        elif X.dtype != self._tensor_dtype:
            X = X.type(self._tensor_dtype)
            if self._device_type == 'cuda' and (not X.is_cuda):
                X = X.to(device=self._device_type)
        assert torch.sum(X<0) == 0, "The input matrix is not non-negative. NMF cannot be applied."

        self.X = X
        self._initialize_W_H()

        for i in range(self._max_iter):
            if self._is_converged():
                self.num_iters = i
                print(f"    Reach convergence after {i+1} iteration(s).")
                break

            # Batch update on H.
            W_t = self.W.T
            X_hat = self._get_X_hat()
            H_factor_numer = (self.X * X_hat.pow(self._beta - 2)) @ W_t
            H_factor_denom = X_hat.pow(self._beta - 1) @ W_t
            self.H *= (H_factor_numer / H_factor_denom)

            # Batch update on W.
            H_t = self.H.T
            X_hat = self._get_X_hat()
            W_factor_numer = H_t @ (self.X * X_hat.pow(self._beta - 2))
            W_factor_denom = H_t @ X_hat.pow(self._beta - 1)
            self.W *= (W_factor_numer / W_factor_denom)

            self._update_loss(self._get_X_hat())

            if i == self._max_iter - 1:
                self.num_iters = self._max_iter
                print(f"    Not converged after {self._max_iter} iteration(s).")
    
    def fit_transform(self, X):
        self.fit(X)
        return self.H
