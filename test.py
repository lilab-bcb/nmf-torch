import numpy as np
import sklearn.decomposition as sd
import nmf
import time
import torch

import matplotlib.pyplot as plt


X = np.random.rand(4000, 2000)

#model1 = sd.NMF(n_components=12, init='random', beta_loss='kullback-leibler', tol=1e-4, max_iter=200, random_state=0, solver='mu')
model2 = nmf.NMF(n_components=12, init='random', loss='l2', tol=1e-4, max_iter=200, random_state=0)
model3 = nmf.NMF_alt(torch.tensor(X).T, rank=12, max_iterations=200, tolerance=1e-4, init_method='random')

#ts1_start = time.time()
#W1 = model1.fit_transform(X)
#ts1_end = time.time()
#print(f"Sklearn uses {ts1_end - ts1_start} s, with reconstruction error {model1.reconstruction_err_} after {model1.n_iter_} iteration(s).")

ts2_start = time.time()
Y = model2.fit_transform(X)
ts2_end = time.time()
print(f"Pytorch uses {ts2_end - ts2_start} s, with final loss at {model2._deviance[-1]} after {model2.num_iters} iteration(s).")
print(model2._deviance)

fig = plt.figure()
ax = plt.plot(np.arange(model2.num_iters+1), model2._deviance, 'b-')
fig.savefig("deviance.pdf", dpi=300)

#ts3_start = time.time()
#model3.fit(beta=2)
#ts3_end = time.time()
#err3 = nmf.NMF._loss(model3._V, model3.reconstruction, beta=2)
#print(f"NMF_CPU uses {ts3_end - ts3_start} s, with final loss at {err3}.")
