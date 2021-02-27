import numpy as np
import sklearn.decomposition as sd
import gensim.models.nmf as gm
import nmf
import time
import torch

from scipy.sparse import csc_matrix
from termcolor import cprint

EPSILON = torch.finfo(torch.float32).eps

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

    p = zip(range(X.shape[1]), map(lambda n: 'gene'+str(n), range(X.shape[1])))
    id2word = {}
    for key, val in p:
        id2word[key] = val
    model0 = gm.Nmf(num_topics=k, id2word=id2word, passes=1, chunksize=1000)
    model1 = sd.NMF(n_components=k, init=init, beta_loss=loss, tol=tol, max_iter=max_iter, random_state=random_state, solver='mu',
                    alpha=alpha, l1_ratio=l1_ratio)
    model2 = nmf.NMF(n_components=k, init=init, beta_loss=loss, tol=tol, max_iter=max_iter, random_state=random_state, update_method='batch',
                     alpha_H=alpha, l1_ratio_H=l1_ratio, alpha_W=alpha, l1_ratio_W=l1_ratio, online_chunk_size=chunk_size)

    cprint("Gensim:", 'green')
    ts_start = time.time()
    model0.update(csc_matrix(X.T))
    ts_end = time.time()
    print(f"Gensim uses {ts_end - ts_start} s.")

    cprint("Sklearn:", 'green')
    print(model1)
    ts_start = time.time()
    W1 = model1.fit_transform(X)
    ts_end = time.time()
    H1 = model1.components_
    err1 = beta_loss(torch.tensor(X), torch.tensor(W1 @ H1), torch.tensor(W1), torch.tensor(H1),
                         l1_reg_H=alpha*l1_ratio, l2_reg_H=alpha*(1-l1_ratio),
                         l1_reg_W=alpha*l1_ratio, l2_reg_W=alpha*(1-l1_ratio),
                         beta=beta, epsilon=EPSILON, square_root=True)
    print(f"Sklearn uses {ts_end - ts_start} s, with reconstruction error {err1} after {model1.n_iter_} iteration(s).")
    print(f"H has {np.sum(W1!=0)} non-zero elements, W has {np.sum(H1!=0)} non-zero elements.")

    cprint("NMF-torch:", 'green')
    ts_start = time.time()
    H2 = model2.fit_transform(X)
    ts_end = time.time()
    W2 = model2.W
    print(model2.reconstruction_err)
    err2 = beta_loss(torch.tensor(X), H2 @ W2, H2, W2,
                     l1_reg_H=alpha*l1_ratio, l2_reg_H=alpha*(1-l1_ratio),
                     l1_reg_W=alpha*l1_ratio, l2_reg_W=alpha*(1-l1_ratio),
                     beta=beta, epsilon=EPSILON, square_root=True)
    print(f"NMF-torch uses {ts_end - ts_start} s, with final loss at {err2} after {model2.num_iters} pass(es).")
    print(f"H has {torch.sum(model2.H!=0).tolist()} non-zero elements, W has {torch.sum(model2.W!=0).tolist()} non-zero elements.")

if __name__ == '__main__':
    cprint("Test 1:", 'yellow')
    run_test("tests/data/nmf_test_1.npy", k=20, loss='frobenius', init='nndsvdar', max_iter=500, chunk_size=5000)

    cprint("Test 2:", 'yellow')
    run_test("tests/data/nmf_test_2.npy", k=20, loss='frobenius', init='nndsvdar', max_iter=500, chunk_size=2000)
