import torch


class NMFBase:
    def __init__(
        self,
        n_components: int,
        init,
        beta_loss: float,
        tol: float,
        random_state: int,
        alpha_W: float,
        l1_ratio_W: float,
        alpha_H: float,
        l1_ratio_H: float,
        fp_precision: str,
        device_type: str,
    ):
        self.k = n_components
        self._beta = beta_loss
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

        self._epsilon = 1e-20

        self._device_type = device_type

        self._init_method = init
        self._tol = tol
        self._random_state = random_state


    @property
    def reconstruction_err(self):
        return self._cur_err


    def _get_HW(self):
        return self.H @ self.W


    def _get_regularization_loss(self, mat, l1_reg, l2_reg):
        res = 0.
        if l1_reg > 0:
            res += l1_reg * mat.norm(p=1)
        if l2_reg > 0:
            res += l2_reg * mat.norm(p=2)**2 / 2
        return res


    def _is_converged(self, prev_err, cur_err, init_err):
        if torch.abs((prev_err - cur_err) / init_err) < self._tol:
            return True
        else:
            return False


    def _add_regularization_terms(self, mat, numer_mat, denom_mat, l1_reg, l2_reg):
        if l1_reg > 0:
            if self._beta <= 1:
                denom_mat += l1_reg
            else:
                numer_mat -= l1_reg
                numer_mat[numer_mat < 0] = 0

        if l2_reg > 0:
            denom_mat += l2_reg * mat


    def _update_matrix(self, mat, numer, denom):
        mat *= (numer / denom)
        mat[denom < self._epsilon] = 0.0


    def _get_regularization_loss(self, mat, l1_reg, l2_reg):
        res = 0.
        if l1_reg > 0:
            res += l1_reg * mat.norm(p=1)
        if l2_reg > 0:
            res += l2_reg * mat.norm(p=2)**2 / 2
        return res


    def _loss(self, square_root=True):
        if self._beta == 2:
            res = torch.trace(self._WWT @ self._HTH) / 2 - torch.trace(self._H_t @ self._XWT) + self._X_SS_half
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
        res += self._get_regularization_loss(self.H, self._l1_reg_H, self._l2_reg_H)
        res += self._get_regularization_loss(self.W, self._l1_reg_W, self._l2_reg_W)

        if square_root:
            return torch.sqrt(2 * res)
        else:
            return res


    def _initialize_H_W(self, eps=1e-6):
        n_samples, n_features = self.X.shape
        if self._init_method is None:
            if self.k < min(n_samples, n_features):
                self._init_method = 'nndsvdar'
            else:
                self._init_method = 'random'

        if self._init_method in ['nndsvd', 'nndsvda', 'nndsvdar']:
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
                H[H == 0] = avg / 100 * torch.rand(H[H==0].shape, dtype=self._tensor_dtype, device=self._device_type)
                W[W == 0] = avg / 100 * torch.rand(W[W==0].shape, dtype=self._tensor_dtype, device=self._device_type)
        elif self._init_method == 'random':
            avg = torch.sqrt(self.X.mean() / self.k)
            H = torch.abs(avg * torch.randn((self.X.shape[0], self.k), dtype=self._tensor_dtype, device=self._device_type))
            W = torch.abs(avg * torch.randn((self.k, self.X.shape[1]), dtype=self._tensor_dtype, device=self._device_type))
        else:
            raise ValueError(f"Invalid init parameter. Got {self._init_method}, but require one of (None, 'nndsvd', 'nndsvda', 'nndsvdar', 'random').")

        self.H = H
        self.W = W

        if self._beta == 2:
            self._W_t = self.W.T
            self._WWT = self.W @ self._W_t
            self._H_t = self.H.T
            self._HTH = self._H_t @ self.H
            self._XWT = self.X @ self._W_t

        self._init_err = self._loss()
        self._prev_err = self._init_err


    def fit(self, X):
        torch.manual_seed(self._random_state)

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
