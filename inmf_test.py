import time
import torch

import numpy as np
import pandas as pd

from nmf import INMFBatch

X = np.load("tests/inmf_data/counts.npy")
df = pd.read_csv("tests/inmf_data/metadata.csv", header=None)
df[0] = df[0].astype('category')

mats = []
for chan in df[0].cat.categories:
    x = X[df.loc[df[0]==chan].index, :].copy()
    mats.append(torch.tensor(x, dtype=torch.float))

print("Start iNMF...")
model = INMFBatch(n_components=20)
ts_start = time.time()
model.fit(mats)
ts_end = time.time()
print(f"iNMF batch finishes in {ts_end - ts_start} s, with error {model.reconstruction_err} after {model.num_iters} iteration(s).")
