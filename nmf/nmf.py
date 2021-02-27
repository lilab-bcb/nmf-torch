import torch
from typing import Optional, Union

class NMF:
    def __init__(
        self,
        n_components: int,
        init: str = "nndsvd",
        beta_loss: Union[str, float] = "frobenius",
        update_method: str = "batch",
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: int = 0,
        alpha_W: float = 0.0,
        l1_ratio_W: float = 0.0,
        alpha_H:float = 0.0,
        l1_ratio_H:float = 0.0,
        fp_precision: str = 'float',
        online_chunk_size: int = 2000,
        online_w_max_iter: int = 200,
        online_h_max_iter: int = 50,
        use_gpu: bool = False,
    ):
        self.k = n_components

        if beta_loss == 'frobenius':
            self._beta = 2
        elif beta_loss == 'kullback-leibler':
            self._beta = 1
        elif beta_loss == 'itakura-saito':
            self._beta = 0
        elif isinstance(beta_loss, int) or isinstance(beta_loss, float):
            self._beta = beta_loss
        else:
            raise ValueError("beta_loss must be a valid value: either from ['frobenius', 'kullback-leibler', 'itakura-saito'], or a numeric value.")

        if update_method in ['online', 'batch']:
            self._update_method = update_method
        else:
            raise ValueError("method must be a valid value from ['online', 'batch'].")

        self._l1_reg_H = alpha_H * l1_ratio_H
        self._l2_reg_H = alpha_H * (1 - l1_ratio_H)
        self._l1_reg_W = alpha_W * l1_ratio_W
        self._l2_reg_W = alpha_W * (1 - l1_ratio_W)

        if (self._beta > 1 and self._beta < 2) and (self._l1_reg_H > 0 or self._l1_reg_W > 0):
            print("L1 norm doesn't have a closed form solution when 1 < beta < 2. Ignore L1 regularization.")
            self._l1_reg_H = 0
            self._l1_reg_W = 0

        if self._beta != 2 and (self._l2_reg_H > 0 or self._l2_reg_W > 0):
            print("L2 norm doesn't have a closed form solution when beta != 2. Ignore L2 regularization.")
            self._l2_reg_H = 0
            self._l2_reg_W = 0

        if fp_precision == 'float':
            self._tensor_dtype = torch.float
        elif fp_precision == 'double':
            self._tensor_dtype = torch.double
        else:
            self._tensor_dtype = fp_precision

        self._epsilon = torch.finfo(self._tensor_dtype).eps
        self._chunk_size = online_chunk_size
        self._w_max_iter = online_w_max_iter
        self._h_max_iter = online_h_max_iter

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

    @property
    def reconstruction_err(self):
        return self._cur_err

    def _loss(self, square_root=False):
        if self._beta == 2:
            res = torch.sum(torch.trace(self._WWT @ self._HTH) / 2 - torch.trace(self._H_t @ self._XWT)) + self._X_SS_half
        elif self._beta == 0 or self._beta == 1:
            Y = self._get_HW()
            X_flat = self.X.flatten()
            Y_flat = Y.flatten()

            idx = X_flat > self._epsilon
            X_flat = X_flat[idx]
            Y_flat = Y_flat[idx]

            # Avoid division by zero
            Y_flat[Y_flat == 0] = self._epsilon

            x_div_y = X_flat / Y_flat
            if self._beta == 0:
                res = x_div_y.sum() - x_div_y.log().sum() - self.X.shape.numel()
            else:
                res = X_flat @ x_div_y.log() - X_flat.sum() + Y.sum()
        else:
            Y = self._get_HW()
            res = torch.sum(self.X.pow(self._beta) - self._beta * self.X * Y.pow(self._beta - 1) + (self._beta - 1) * Y.pow(self._beta))
            res /= (self._beta * (self._beta - 1))

        # Add regularization terms.
        if self._l1_reg_H > 0:
            res += self._l1_reg_H * self.H.norm(p=1)
        if self._l2_reg_H > 0:
            res += self._l2_reg_H * self.H.norm(p=2)**2 / 2
        if self._l1_reg_W > 0:
            res += self._l1_reg_W * self.W.norm(p=1)
        if self._l2_reg_W > 0:
            res += self._l2_reg_W * self.W.norm(p=2)**2 / 2

        if square_root:
            return torch.sqrt(2 * res)
        else:
            return res

    def _is_converged(self, prev_err, cur_err, init_err):
        if torch.abs((prev_err - cur_err) / init_err) < self._tol:
            return True
        else:
            return False

    def _get_HW(self):
        return self.H @ self.W

    def _initialize_H_W(self, eps=1e-6):
        n_samples, n_features = self.X.shape
        if self._init_method is None:
            if self.k < min(n_samples, n_features):
                self._init_method = 'nndsvdar'
            else:
                self._init_method = 'random'

        if self._init_method in ['nndsvd', 'nndsvda', 'nndsvdar']:
            torch.manual_seed(self._random_state)
            U, S, V = torch.svd_lowrank(self.X, q=self.k)

            H= torch.zeros_like(U, dtype=self._tensor_dtype, device=self._device_type)
            W = torch.zeros_like(V.T, dtype=self._tensor_dtype, device=self._device_type)
            H[:, 0] = S[0].sqrt() * U[:, 0]
            W[0, :] = S[0].sqrt() * V[:, 0]

            for j in range(2, self.k):
                x, y = U[:, j], V[:, j]
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
                W[j, :] = factor * v

            H[H < eps] = 0
            W[W < eps] = 0

            if self._init_method == 'nndsvda':
                avg = self.X.mean()
                H[H == 0] = avg
                W[W == 0] = avg
            elif self._init_method == 'nndsvdar':
                avg = self.X.mean()
                rng = torch.Generator(device=self._device_type).manual_seed(self._random_state)
                H[H == 0] = avg / 100 * torch.rand(H[H==0].shape, generator=rng, device=self._device_type)
                W[W == 0] = avg / 100 * torch.rand(W[W==0].shape, generator=rng, device=self._device_type)
        elif self._init_method == 'random':
            avg = torch.sqrt(self.X.mean() / self.k)
            rng = torch.Generator(device=self._device_type).manual_seed(self._random_state)
            H = torch.abs(avg * torch.randn((self.X.shape[0], self.k), generator=rng, dtype=self._tensor_dtype, device=self._device_type))
            W = torch.abs(avg * torch.randn((self.k, self.X.shape[1]), generator=rng, dtype=self._tensor_dtype, device=self._device_type))
        else:
            raise ValueError(f"Invalid init parameter. Got {self._init_method}, but require one of (None, 'nndsvd', 'nndsvda', 'nndsvdar', 'random').")

        self.H = H
        self.W = W

        if self._update_method == 'batch' and self._beta == 2:
            self._W_t = self.W.T
            self._WWT = self.W @ self._W_t
            self._H_t = self.H.T
            self._HTH = self._H_t @ self.H
            self._XWT = self.X @ self._W_t

        self._init_err = self._loss(square_root=True)
        self._prev_err = self._init_err

    def _add_regularization_terms(self, mat, numer_mat, denom_mat, l1_reg, l2_reg):
        if l1_reg > 0:
            if self._beta <= 1:
                denom_mat += l1_reg
            else:
                numer_mat -= l1_reg
                numer_mat[numer_mat < 0] = 0

        if l2_reg > 0:
            denom_mat += l2_reg * mat

    def _W_err(self, A, B, WWT=None):
        W_t = self.W.T
        if WWT is None:
            WWT = self.W @ W_t

        return torch.sum(torch.trace(WWT @ A) / 2 - torch.trace(B @ W_t))

    def _h_err(self, h_t, hth, WWT, xWT):
        return torch.sum(torch.trace(WWT @ hth) / 2 - torch.trace(h_t @ xWT))

    def _online_update_one_pass(self):
        indices = torch.randperm(self.X.shape[0], device=self._device_type)
        A = torch.zeros((self.k, self.k), dtype=self._tensor_dtype, device=self._device_type)
        B = torch.zeros((self.k, self.X.shape[1]), dtype=self._tensor_dtype, device=self._device_type)

        i = 0
        num_processed = 0
        cnt = 0
        import time
        while i < indices.shape[0]:
            cnt += 1
            idx = indices[i:(i+self._chunk_size)]
            cur_chunksize = idx.shape[0]
            x = self.X[idx, :]
            h = self.H[idx, :]

            # Online update H.
            W_t = self.W.T
            WWT = self.W @ W_t
            xWT = x @ W_t
            h_factor_numer = xWT.clone()
            h_t = h.T
            hth = h_t @ h

            prev_h_err = self._h_err(h_t, hth, WWT, xWT)

            for j in torch.arange(self._h_max_iter):
                h_factor_denom = h @ WWT

                self._add_regularization_terms(h, h_factor_numer, h_factor_denom, self._l1_reg_H, self._l2_reg_H)
                h_factor_denom[h_factor_denom == 0] = self._epsilon
                h *= (h_factor_numer / h_factor_denom)
                h_t = h.T
                hth = h_t @ h

                cur_h_err = self._h_err(h_t, hth, WWT, xWT)
                if self._is_converged(prev_h_err, cur_h_err, prev_h_err) or (j + 1 == self._h_max_iter):
                    break
                else:
                    prev_h_err = cur_h_err

            self.H[idx, :] = h

            # Update sufficient statistics A and B.
            num_after = num_processed + cur_chunksize

            A *= num_processed
            A += hth
            A /= num_after

            B *= num_processed
            B += h_t @ x
            B /= num_after

            num_processed = num_after

            # Online update W.
            W_factor_numer = B
            prev_W_err = self._W_err(A, B, WWT)

            for j in torch.arange(self._w_max_iter):
                W_factor_denom = A @ self.W

                self._add_regularization_terms(self.W, W_factor_numer, W_factor_denom, self._l1_reg_W, self._l2_reg_W)
                W_factor_denom[W_factor_denom == 0] = self._epsilon
                self.W *= (W_factor_numer / W_factor_denom)

                cur_W_err = self._W_err(A, B)
                if self._is_converged(prev_W_err, cur_W_err, prev_W_err) or (j + 1 == self._w_max_iter):
                    break
                else:
                    prev_W_err = cur_W_err

            i += self._chunk_size


    def _online_update_H(self, W_t=None, WWT=None):
        if W_t is None:
            W_t = self.W.T

        if WWT is None:
            WWT = self.W @ W_t

        i = 0
        sum_h_err = 0.
        while i < self.H.shape[0]:
            x = self.X[i:(i+self._chunk_size), :]
            h = self.H[i:(i+self._chunk_size), :]

            xWT = x @ W_t
            h_factor_numer = xWT.clone()
            h_t = h.T
            hth = h_t @ h
            prev_h_err = self._h_err(h_t, hth, WWT, xWT)

            for j in torch.arange(self._h_max_iter):
                h_factor_denom = h @ WWT
                self._add_regularization_terms(h, h_factor_numer, h_factor_denom, self._l1_reg_H, self._l2_reg_H)
                h_factor_denom[h_factor_denom == 0] = self._epsilon
                h *= (h_factor_numer / h_factor_denom)
                h_t = h.T
                hth = h_t @ h
                cur_h_err = self._h_err(h_t, hth, WWT, xWT)
                if self._is_converged(prev_h_err, cur_h_err, prev_h_err):
                    break
                elif j + 1 == self._h_max_iter:
                    break
                else:
                    prev_h_err = cur_h_err

            sum_h_err += cur_h_err
            i += self._chunk_size

        return sum_h_err


    def _online_update_H_W(self):
        self._chunk_size = min(self.X.shape[0], self._chunk_size)

        for i in range(self._max_iter):
            self._online_update_one_pass()

            # Update H again at the end of each pass.
            H_err = self._online_update_H()

            self._cur_err = torch.sqrt(2 * (H_err + self._X_SS_half))
            if self._is_converged(self._prev_err, self._cur_err, self._init_err):
                self.num_iters = i + 1
                break
            elif i == self._max_iter - 1:
                self.num_iters = self._max_iter
                print(f"    Not converged after {self._max_iter} pass(es).")
            else:
                self._prev_err = self._cur_err


    def _batch_update_H(self):
        if self._beta == 2:
            H_factor_numer = self._XWT.clone()
            H_factor_denom = self.H @ self._WWT
        else:
            HW = self._get_HW()
            HW_pow = HW.pow(self._beta - 2)
            H_factor_numer = (self.X * HW_pow) @ self._W_t
            H_factor_denom = (HW_pow * HW) @ self._W_t

        self._add_regularization_terms(self.H, H_factor_numer, H_factor_denom, self._l1_reg_H, self._l2_reg_H)
        H_factor_denom[H_factor_denom == 0] = self._epsilon
        self.H *= (H_factor_numer / H_factor_denom)

        if self._beta == 2:
            self._H_t = self.H.T
            self._HTH = self._H_t @ self.H

    def _batch_update_W(self):
        if self._beta == 2:
            W_factor_numer = self._H_t @ self.X
            W_factor_denom = self._HTH @ self.W
        else:
            H_t = self.H.T
            HW = self._get_HW()
            HW_pow = HW.pow(self._beta - 2)
            W_factor_numer = H_t @ (self.X * HW_pow)
            W_factor_denom = H_t @ (HW_pow * HW)

        self._add_regularization_terms(self.W, W_factor_numer, W_factor_denom, self._l1_reg_W, self._l2_reg_W)
        W_factor_denom[W_factor_denom == 0] = self._epsilon
        self.W *= (W_factor_numer / W_factor_denom)

        if self._beta == 2:
            self._W_t = self.W.T
            self._WWT = self.W @ self._W_t
            self._XWT = self.X @ self._W_t

    def _batch_update_H_W(self):
        for i in range(self._max_iter):
            if (i + 1) % 10 == 0:
                self._cur_err = self._loss(square_root=True)
                if self._is_converged(self._prev_err, self._cur_err, self._init_err):
                    self.num_iters = i + 1
                    break
                else:
                    self._prev_err = self._cur_err

            self._batch_update_H()
            self._batch_update_W()

            if i == self._max_iter - 1:
                self.num_iters = self._max_iter
                print(f"    Not converged after {self._max_iter} iteration(s).")

    @torch.no_grad()
    def fit(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self._tensor_dtype, device=self._device_type)
        else:
            if X.dtype != self._tensor_dtype:
                X = X.type(self._tensor_dtype)
            if self._device_type == 'cuda' and (not X.is_cuda):
                X = X.to(device=self._device_type)
        assert torch.sum(X<0) == 0, "The input matrix is not non-negative. NMF cannot be applied."

        self.X = X
        if self._beta == 2:  # Cache sum of X^2 divided by 2 for speed-up of calculating beta loss.
            self._X_SS_half = torch.sum(X**2) / 2
        self._initialize_H_W()

        if self._beta == 2 and self._update_method == 'online':
            self._online_update_H_W()
        else:
            if self._update_method == 'online':
                print("Cannot perform online update when beta not equal to 2. Switch to batch update method.")
            self._batch_update_H_W()

    def fit_transform(self, X):
        self.fit(X)
        return self.H
