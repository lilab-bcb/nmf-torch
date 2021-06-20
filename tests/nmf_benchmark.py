import numpy as np
import sklearn.decomposition as sd
import time
import torch

from termcolor import cprint
from nmf import run_nmf

EPSILON = 1e-20

def beta_loss(X, Y, H, W, beta, epsilon, l1_reg_H=0., l2_reg_H=0., l1_reg_W=0., l2_reg_W=0., square_root=False):
        if beta == 2:
            res = torch.sum((X - Y)**2) / 2
        if beta == 0 or beta == 1:
            X_flat = X.flatten()
            Y_flat = Y.flatten()

            idx = X_flat > epsilon
            X_flat = X_flat[idx]
            Y_flat = Y_flat[idx]

            # Avoid division by zero
            Y_flat[Y_flat == 0] = epsilon

            x_div_y = X_flat / Y_flat
            if beta == 0:
                res = x_div_y.sum() - x_div_y.log().sum() - X.shape.numel()
            else:
                res = X_flat @ x_div_y.log() - X_flat.sum() + Y.sum()
        else:
            res = torch.sum(X.pow(beta) - beta * X * Y.pow(beta - 1) + (beta - 1) * Y.pow(beta))
            res /= (beta * (beta - 1))

        # Add regularization terms.
        res += l1_reg_H * H.norm(p=1) + l2_reg_H * H.norm(p=2)**2 / 2
        res += l1_reg_W * W.norm(p=1) + l2_reg_W * W.norm(p=2)**2 / 2

        if square_root:
            return torch.sqrt(2 * res)
        else:
            return res

def run_test(filename, k, init='nndsvdar', loss='kullback-leibler', tol=1e-4, max_iter=200, random_state=0,
             alpha=0.0, l1_ratio=0.0, chunk_size=2000):
    X = np.load(filename)
    print(X.shape)

    if loss == 'kullback-leibler':
        beta = 1
    elif loss == 'frobenius':
        beta = 2
    elif loss == 'itakura-saito':
        beta = 0
    else:
        raise ValueError("Beta loss not supported!")



    cprint("Sklearn:", 'green')
    model0 = sd.NMF(n_components=k, init=init, beta_loss=loss, tol=tol, max_iter=max_iter, random_state=random_state, solver='mu',
                    alpha=alpha, l1_ratio=l1_ratio)
    print(model0)
    ts_start = time.time()
    W0 = model0.fit_transform(X)
    ts_end = time.time()
    H0 = model0.components_
    err0 = beta_loss(torch.tensor(X), torch.tensor(W0 @ H0), torch.tensor(W0), torch.tensor(H0),
                         l1_reg_H=alpha*l1_ratio, l2_reg_H=alpha*(1-l1_ratio),
                         l1_reg_W=alpha*l1_ratio, l2_reg_W=alpha*(1-l1_ratio),
                         beta=beta, epsilon=EPSILON, square_root=True)
    print(f"Sklearn uses {ts_end - ts_start} s, with reconstruction error {err0} after {model0.n_iter_} iteration(s).")
    print(f"H has {np.sum(W0!=0)} non-zero elements, W has {np.sum(H0!=0)} non-zero elements.")

    cprint("NMF-torch batch:", 'green')
    ts_start = time.time()
    H1, W1, _ = run_nmf(X, n_components=k, init=init, beta_loss=loss, update_method='batch', max_iter=max_iter, tol=tol,
                        random_state=random_state, alpha_W=alpha, l1_ratio_W=l1_ratio, alpha_H=alpha, l1_ratio_H=l1_ratio)
    ts_end = time.time()
    err1 = beta_loss(torch.tensor(X), torch.tensor(H1 @ W1), torch.tensor(H1), torch.tensor(W1),
                     l1_reg_H=alpha*l1_ratio, l2_reg_H=alpha*(1-l1_ratio),
                     l1_reg_W=alpha*l1_ratio, l2_reg_W=alpha*(1-l1_ratio),
                     beta=beta, epsilon=EPSILON, square_root=True)
    print(f"NMF-torch uses {ts_end - ts_start} s, with final loss at {err1}.")
    print(f"H has {np.sum(H1!=0)} non-zero elements, W has {np.sum(W1!=0)} non-zero elements.")

    cprint("NMF-torch online:", 'green')
    ts_start = time.time()
    H2, W2, _ = run_nmf(X, n_components=k, init=init, beta_loss=loss, update_method='online', tol=tol, random_state=random_state,
                        alpha_W=alpha, l1_ratio_W=l1_ratio, alpha_H=alpha, l1_ratio_H=l1_ratio,
                        online_max_pass=max_iter, online_chunk_size=chunk_size)
    ts_end = time.time()
    err2 = beta_loss(torch.tensor(X), torch.tensor(H2 @ W2), torch.tensor(H2), torch.tensor(W2),
                     l1_reg_H=alpha*l1_ratio, l2_reg_H=alpha*(1-l1_ratio),
                     l1_reg_W=alpha*l1_ratio, l2_reg_W=alpha*(1-l1_ratio),
                     beta=beta, epsilon=EPSILON, square_root=True)
    print(f"NMF-torch uses {ts_end - ts_start} s, with final loss at {err2}.")
    print(f"H has {np.sum(H2!=0)} non-zero elements, W has {np.sum(W2!=0)} non-zero elements.")

if __name__ == '__main__':
    cprint("Test 1:", 'yellow')
    run_test("tests/data/nmf_test_1.npy", k=20, loss='frobenius', init='random', max_iter=500, chunk_size=5000)

    cprint("Test 2:", 'yellow')
    run_test("tests/data/nmf_test_2.npy", k=20, loss='frobenius', init='random', max_iter=500, chunk_size=2000)
