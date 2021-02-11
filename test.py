import numpy as np
import sklearn.decomposition as sd
import nmf
import time
import torch

from termcolor import cprint

EPSILON = torch.finfo(torch.float32).eps

def run_test(filename, k, init='nndsvdar', loss='kullback-leibler', tol=1e-4, max_iter=200, random_state=0):
    X = np.load(filename)
    X += 1

    if loss == 'kullback-leibler':
        beta = 1
    elif loss == 'frobenius':
        beta = 2
    elif loss == 'itakura-saito':
        beta = 0
    else:
        raise ValueError("Beta loss not supported!")

    model1 = sd.NMF(n_components=k, init=init, beta_loss=loss, tol=tol, max_iter=max_iter, random_state=random_state, solver='mu')
    model2 = nmf.NMF(n_components=k, init=init, loss=loss, tol=tol, max_iter=max_iter, random_state=random_state)
    model3 = nmf.NMF_alt(torch.tensor(X).T, rank=k, max_iterations=max_iter, test_conv=10, tolerance=tol, init_method=init)

    ts_start = time.time()
    W1 = model1.fit_transform(X)
    ts_end = time.time()
    H1 = model1.components_
    print(f"Sklearn uses {ts_end - ts_start} s, with reconstruction error {model1.reconstruction_err_} after {model1.n_iter_} iteration(s).")
    err1 = nmf.NMF._loss(torch.tensor(X), torch.tensor(W1 @ H1), beta=beta, epsilon=EPSILON, square_root=True)
    print(f"Standardized error = {err1}.")

    ts_start = time.time()
    Y = model2.fit_transform(X)
    ts_end = time.time()
    print(f"NMF-torch uses {ts_end - ts_start} s, with final loss at {model2.reconstruction_err} after {model2.num_iters} iteration(s).")

    ts_start = time.time()
    model3.fit(beta=beta)
    ts_end = time.time()
    print(f"NMF_CPU uses {ts_end - ts_start} s, with final loss at {model3._kl_loss} after {model3._iter + 1} iteration(s).")
    err3 = nmf.NMF._loss(model3._V, model3.reconstruction, beta=beta, epsilon=EPSILON, square_root=True)
    print(f"Standardized erro = {err3}.")

if __name__ == '__main__':
    cprint("Test 1:", 'yellow')
    run_test("tests/data/nmf_test_1.npy", k=12, loss='frobenius')

    cprint("Test 2:", 'yellow')
    run_test("tests/data/nmf_test_2.npy", k=12)
