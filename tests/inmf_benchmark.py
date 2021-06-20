import time
import torch

#import pegasus as pg
import numpy as np
import pandas as pd

from nmf import integrative_nmf

def loss(X, H, W, V, lam, dtype='double'):
    n_batches = len(X)
    for k in range(n_batches):
        H[k] = torch.tensor(H[k])
        V[k] = torch.tensor(V[k])

    res = 0.0
    for k in range(len(X)):
        res += torch.norm(X[k].double() - H[k].double() @ (W.double() + V[k].double()), p=2)**2
        if lam > 0:
            res += lam * torch.norm(H[k].double() @ V[k].double(), p=2)**2

    return torch.sqrt(res)


def run_test(mats, algo, mode, n_components, lam, n_jobs, seed, fp_precision, batch_max_iter):
    print(f"{algo} {mode} Experiment...")

    ts_start = time.time()
    H, W, V, err = integrative_nmf(mats, algo=algo, mode=mode, n_components=n_components, lam=lam,
                        n_jobs=n_jobs, random_state=seed, fp_precision=fp_precision, batch_max_iter=batch_max_iter)
    ts_end = time.time()
    err_confirm = loss(mats, H, torch.tensor(W), V, lam)
    print(f"{algo} {mode} finishes in {ts_end - ts_start} s, with error {err} (confirmed with {err_confirm}).")

#data = pg.read_input("MantonBM_nonmix.zarr.zip")
#pg.qc_metrics(data, min_genes=500, max_genes=6000, mito_prefix='MT-', percent_mito=10)
#pg.filter_data(data)
#pg.identify_robust_genes(data)
#pg.log_norm(data)
#pg.highly_variable_features(data, consider_batch=True)
#keyword = pg.select_features(data, features='highly_variable_features', standardize=True, max_value=10)
#X = (data.uns[keyword] + data.uns['stdzn_mean'] / data.uns['stdzn_std']).astype(np.float32)
#X[X < 0] = 0.0
#np.save("inmf_data/counts.npy", X)
#data.obs[['Channel']].to_csv("inmf_data/metadata.csv", index=False, header=False)
X = np.load("inmf_data/subset/counts.npy")

df = pd.read_csv("inmf_data/subset/metadata.csv", header=None)
df[0] = df[0].astype('category')
mats = []
#for chan in data.obs['Channel'].cat.categories:
#    x = X[df_obs.loc[df_obs['Channel']==chan].index, :].copy()
#    mats.append(torch.tensor(x))
for chan in df[0].cat.categories:
    x = X[df.loc[df[0]==chan].index, :].copy()
    print(x.shape)
    mats.append(torch.tensor(x, dtype=torch.float))

print("Start iNMF...")
#rnd_seeds = [28728712, 39074257, 751935947, 700933753, 1315698701, 1096583738, 1381716902, 1862944882, 472642840, 530691960]
rnd_seeds = [0]
#rnd_seeds = [3365, 2217, 629, 715, 4289, 3849, 625, 6598, 8275, 9570]

cnt = 0
lam = 5.0
n_jobs = 12
algo_list = ['hals', 'bpp', 'mu']
mode_list = ['batch', 'online']
for seed in rnd_seeds:
    cnt += 1
    print(f"{cnt}. Experiment with random seed {seed}...")

    for algo in algo_list:
        for mode in mode_list:
            run_test(mats, algo, mode, n_components=20, lam=lam, n_jobs=n_jobs, seed=seed, fp_precision='float', batch_max_iter=500)
