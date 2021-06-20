from ._nmf_batch_base import NMFBatchBase


class NMFBatchMU(NMFBatchBase):
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

    def _get_HW(self):
        return self.H @ self.W


    def _update_H(self):
        if self._beta == 2:
            H_factor_numer = self._XWT.clone()
            H_factor_denom = self.H @ self._WWT
        else:
            HW = self._get_HW()
            HW_pow = HW.pow(self._beta - 2)
            H_factor_numer = (self.X * HW_pow) @ self.W.T
            H_factor_denom = (HW_pow * HW) @ self.W.T

        self._add_regularization_terms(self.H, H_factor_numer, H_factor_denom, self._l1_reg_H, self._l2_reg_H)
        self._update_matrix(self.H, H_factor_numer, H_factor_denom)

        if self._beta == 2:
            self._HTH = self.H.T @ self.H


    def _update_W(self):
        if self._beta == 2:
            W_factor_numer = self.H.T @ self.X
            W_factor_denom = self._HTH @ self.W
        else:
            HW = self._get_HW()
            HW_pow = HW.pow(self._beta - 2)
            W_factor_numer = self.H.T @ (self.X * HW_pow)
            W_factor_denom = self.H.T @ (HW_pow * HW)

        self._add_regularization_terms(self.W, W_factor_numer, W_factor_denom, self._l1_reg_W, self._l2_reg_W)
        self._update_matrix(self.W, W_factor_numer, W_factor_denom)

        if self._beta == 2:
            self._WWT = self.W @ self.W.T
            self._XWT = self.X @ self.W.T


    def fit(self, X):
        super().fit(X)

        # Batch update.
        for i in range(self._max_iter):
            self._update_H()
            self._update_W()

            if (i + 1) % 10 == 0:
                self._cur_err = self._loss()
                print(f" niter={i+1}, loss={self._cur_err}.")
                if self._is_converged(self._prev_err, self._cur_err, self._init_err):
                    self.num_iters = i + 1
                    print(f"    Converged after {self.num_iters} iteration(s).")
                    return

                self._prev_err = self._cur_err

        self.num_iters = self._max_iter
        print(f"    Not converged after {self.num_iters} iteration(s).")
