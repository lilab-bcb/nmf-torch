import torch

class NMF:
    def __init__(self, n_components, init='nndsvd', loss='frobenius', 
                 beta = 2, max_iter=200, tol=1e-4, random_state=0, 
                 alpha_W=0.0, l1_ratio_W=0.0, alpha_H=0.0, l1_ratio_H=0.0,
                 fp_precision='float', use_gpu=False):
        self.k = n_components

        if loss == 'frobenius':
            self._beta = 2
        if loss == 'kullback-leibler':
            self._beta = 1
        elif loss == 'itakura-saito':
            self._beta = 0
        else:
            self._beta = beta

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

    def _loss(self, X, Y, square_root=False):
        if self._beta == 2:
            res = torch.sum((X - Y)**2) / 2
        elif self._beta == 0 or self._beta == 1:
            X_flat = X.flatten()
            Y_flat = Y.flatten()

            idx = X_flat > self._epsilon
            X_flat = X_flat[idx]
            Y_flat = Y_flat[idx]

            # Avoid division by zero
            Y_flat[Y_flat == 0] = self._epsilon

            x_div_y = X_flat / Y_flat
            if self._beta == 0:
                res = x_div_y.sum() - x_div_y.log().sum() - X.shape.numel()
            else:
                res = X_flat @ x_div_y.log() - X_flat.sum() + Y.sum()
        else:
            res = torch.sum(X.pow(self._beta) - self._beta * X * Y.pow(self._beta - 1) + (self._beta - 1) * Y.pow(self._beta))
            res /= (self._beta * (self._beta - 1))

        # Add regularization terms.
        res += self._l1_reg_H * self.H.norm(p=1) + self._l2_reg_H * self.H.norm(p=2)**2 / 2
        res += self._l1_reg_W * self.W.norm(p=1) + self._l2_reg_W * self.W.norm(p=2)**2 / 2

        if square_root:
            return torch.sqrt(2 * res)
        else:
            return res
    
    def _is_converged(self):
        if torch.abs((self._prev_err - self._cur_err) / self._init_err) < self._tol:
            return True
        else:
            self._prev_err = self._cur_err
            return False

    def _get_HW(self):
        return self.H @ self.W

    def _initialize_HW(self, eps=1e-6):
        n_samples, n_features = self.X.shape
        if self._init_method is None:
            if self.k < min(n_samples, n_features):
                self._init_method = 'nndsvdar'
            else:
                self._init_method = 'random'
        
        if self._init_method in ['nndsvd', 'nndsvda', 'nndsvdar']:
            torch.Generator(device=self._device_type).manual_seed(self._random_state)
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

        self._init_err = self._loss(self.X, self._get_HW(), square_root=True)
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

    def _online_update_H(self):
        W_t = self.W.T
        WWT = self.W @ W_t
        H_factor_numer = self.X @ W_t
        H_factor_denom = self.H @ WWT

        self._add_regularization_terms(self.H, H_factor_numer, H_factor_denom, self._l1_reg_H, self._l2_reg_H)
        H_factor_denom[H_factor_denom == 0] = self._epsilon
        self.H *= (H_factor_numer / H_factor_denom)

    def _batch_update_H(self):
        W_t = self.W.T
        HW = self._get_HW()
        HW_pow = HW.pow(self._beta - 2)
        H_factor_numer = (self.X * HW_pow) @ W_t
        H_factor_denom = (HW_pow * HW) @ W_t

        self._add_regularization_terms(self.H, H_factor_numer, H_factor_denom, self._l1_reg_H, self._l2_reg_H)
        H_factor_denom[H_factor_denom == 0] = self._epsilon
        self.H *= (H_factor_numer / H_factor_denom)

    def _online_update_W(self):
        H_t = self.H.T
        HTH = H_t @ self.H
        W_factor_numer = H_t @ self.X
        W_factor_denom = HTH @ self.W

        self._add_regularization_terms(self.W, W_factor_numer, W_factor_denom, self._l1_reg_W, self._l2_reg_W)
        W_factor_denom[W_factor_denom == 0] = self._epsilon
        self.W *= (W_factor_numer / W_factor_denom)

    def _batch_update_W(self):
        H_t = self.H.T
        HW = self._get_HW()
        HW_pow = HW.pow(self._beta - 2)
        W_factor_numer = H_t @ (self.X * HW_pow)
        W_factor_denom = H_t @ (HW_pow * HW)

        self._add_regularization_terms(self.W, W_factor_numer, W_factor_denom, self._l1_reg_W, self._l2_reg_W)
        W_factor_denom[W_factor_denom == 0] = self._epsilon
        self.W *= (W_factor_numer / W_factor_denom) 

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
        self._initialize_HW()

        for i in range(self._max_iter):
            if (i + 1) % 10 == 0:
                self._cur_err = self._loss(self.X, self._get_HW(), square_root=True)
                if self._is_converged():
                    self.num_iters = i + 1
                    print(f"    Reach convergence after {i+1} iteration(s).")
                    break

            # Update H.
            if self._beta == 2:
                self._online_update_H()
            else:
                self._batch_update_H()

            # Update W.
            if self._beta == 2:
                self._online_update_W()
            else:
                self._batch_update_W()         

            if i == self._max_iter - 1:
                self.num_iters = self._max_iter
                print(f"    Not converged after {self._max_iter} iteration(s).")
    
    def fit_transform(self, X):
        self.fit(X)
        return self.H
