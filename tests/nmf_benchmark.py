import numpy as np
import sklearn.decomposition as sd
import time, random, gc
import torch

from termcolor import cprint
from nmf import run_nmf
from threadpoolctl import threadpool_limits

EPSILON = 1e-20

def beta_loss(X, Y, H, W, beta, epsilon, l1_reg_H=0., l2_reg_H=0., l1_reg_W=0., l2_reg_W=0., dtype='double'):
    if dtype == 'double':
        X = X.double()
        Y = Y.double()
        H = H.double()
        W = W.double()

    if beta == 2:
        if dtype == 'double':
            res = torch.sum((X.double() - Y.double())**2) / 2
        else:
            res = torch.sum((X.float() - Y.float())**2) / 2

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


    return torch.sqrt(2 * res)


def run_test(filename, algo, mode, k, n_jobs, fp=None, init='nndsvdar', loss='frobenius', tol=1e-4, max_iter=200, random_state=0,
             alpha=0.0, l1_ratio=0.0, chunk_size=5000):
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

    #if method == 'sklearn mu':
    #    model = sd.NMF(n_components=k, init=init, beta_loss=loss, tol=tol, max_iter=max_iter, random_state=random_state, solver='mu',
    #                alpha=alpha, l1_ratio=l1_ratio)
    #    with threadpool_limits(limits=n_jobs):
    #        ts_start = time.time()
    #        W1 = model.fit_transform(X)
    #        ts_end = time.time()
    #    H1 = model.components_
    #    err = beta_loss(torch.tensor(X), torch.tensor(W1 @ H1), torch.tensor(W1), torch.tensor(H1),
    #                     l1_reg_H=alpha*l1_ratio, l2_reg_H=alpha*(1-l1_ratio),
    #                     l1_reg_W=alpha*l1_ratio, l2_reg_W=alpha*(1-l1_ratio),
    #                     beta=beta, epsilon=EPSILON)
    #    print(f"H has {np.sum(W1!=0)} non-zero elements, W has {np.sum(H1!=0)} non-zero elements. Iterations: {model.n_iter_}.")
    #elif method == 'sklearn cd':
    #    model = sd.NMF(n_components=k, init=init, beta_loss=loss, tol=tol, max_iter=max_iter, random_state=random_state, solver='cd',
    #                alpha=alpha, l1_ratio=l1_ratio)
    #    with threadpool_limits(limits=n_jobs):
    #        ts_start = time.time()
    #        W1 = model.fit_transform(X)
    #        ts_end = time.time()
    #    H1 = model.components_
    #    err = beta_loss(torch.tensor(X), torch.tensor(W1 @ H1), torch.tensor(W1), torch.tensor(H1),
    #                     l1_reg_H=alpha*l1_ratio, l2_reg_H=alpha*(1-l1_ratio),
    #                     l1_reg_W=alpha*l1_ratio, l2_reg_W=alpha*(1-l1_ratio),
    #                     beta=beta, epsilon=EPSILON)
    #    err_double = beta_loss(torch.tensor(X), torch.tensor(W1 @ H1), torch.tensor(W1), torch.tensor(H1),
    #                     l1_reg_H=alpha*l1_ratio, l2_reg_H=alpha*(1-l1_ratio),
    #                     l1_reg_W=alpha*l1_ratio, l2_reg_W=alpha*(1-l1_ratio),
    #                     beta=beta, epsilon=EPSILON, dtype='double')
    #    print(f"H has {np.sum(W1!=0)} non-zero elements, W has {np.sum(H1!=0)} non-zero elements. Iterations: {model.n_iter_}.")
    ts_start = time.time()
    H, W, err = run_nmf(X, k, init=init, beta_loss=loss, algo=algo, mode=mode, tol=tol, n_jobs=n_jobs, random_state=random_state,
                            alpha_W=alpha, l1_ratio_W=l1_ratio, alpha_H=alpha, l1_ratio_H=l1_ratio, fp_precision='float')
    ts_end = time.time()
    err_confirm = beta_loss(torch.tensor(X), torch.tensor(H @ W), torch.tensor(H), torch.tensor(W), beta=beta, epsilon=EPSILON,
                        l1_reg_H=alpha*l1_ratio, l2_reg_H=alpha*(1-l1_ratio),
                        l1_reg_W=alpha*l1_ratio, l2_reg_W=alpha*(1-l1_ratio))
    print(f"{algo} {mode} takes {ts_end - ts_start} s, with error {err} ({err_confirm} confirmed).")
    if fp is not None:
        fp.write(f"{algo} {mode},{ts_end - ts_start} s,{err}\n")


def run_batch_test(file_in, test_name, k, max_iter, chunk_size=50000, loss='frobenius', init='random'):
    algo_list = ['hals', 'mu', 'bpp']
    mode_list = ['batch', 'online']
    n_jobs = 12

    #log_file = f"{test_name}.log"
    #with open(log_file, 'a') as fout:
    #    fout.write("method,runtime,error\n")

    #rand_ints = [3365, 2217, 629, 715, 4289, 3849, 625, 6598, 8275, 9570]
    rand_ints = [0]
    n_exp = len(rand_ints)
    #n_exp = 10
    #while len(rand_ints) < n_exp:
    #    n = random.randint(0, 10000)
    #    if n not in rand_ints:
    #        rand_ints.append(n)
    print("Random seeds in use:")
    print(rand_ints)

    #with open(log_file, 'a') as fout:
    for i in range(n_exp):
        for algo in algo_list:
            for mode in mode_list:
                cprint(f"{i+1}-th {algo} {mode}:", 'green')
                #run_test(file_in, algo, mode, k=k, n_jobs=n_jobs, fp=fout, loss=loss, init=init, max_iter=max_iter, chunk_size=chunk_size, random_state=rand_ints[i])
                run_test(file_in, algo, mode, k=k, n_jobs=n_jobs, fp=None, loss=loss, init=init, max_iter=max_iter, chunk_size=chunk_size, random_state=rand_ints[i])
                gc.collect()


if __name__ == '__main__':
    #cprint("Test 1:", 'blue')
    #run_batch_test("tests/data/MantonBM.npy", test_name='MantonBM', k=20, max_iter=500)

    #cprint("Test 2:", 'blue')
    #run_batch_test("tests/data/Mouse_neuron.npy", test_name='Mouse_neuron', k=20, max_iter=500)

    cprint("Test 3:", 'blue')
    run_batch_test("data/nmf_test_1.npy", test_name='nmf_test_1', k=20, max_iter=500)
