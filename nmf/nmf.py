import numpy as np
import torch
from typing import Union, Tuple
from ._nmf_batch import NMFBatch
from ._nmf_online import NMFOnline
from ._nmf_batch_hals import NMFBatchHALS
from ._nmf_online_hals import NMFOnlineHALS

def run_nmf(
    X: Union[np.array, torch.tensor],
    n_components: int,
    init: str = "nndsvdar",
    beta_loss: Union[str, float] = "frobenius",
    update_method: str = "batch mu",
    max_iter: int = 200,
    tol: float = 1e-4,
    random_state: int = 0,
    alpha_W: float = 0.0,
    l1_ratio_W: float = 0.0,
    alpha_H: float = 0.0,
    l1_ratio_H: float = 0.0,
    fp_precision: Union[str, torch.dtype] = "float",
    online_max_pass: int = 10,
    online_chunk_size: int = 2000,
    online_w_max_iter: int = 200,
    online_h_max_iter: int = 50,
    use_gpu: bool = False,
) -> Tuple[np.array, np.array, float]:
    """
    Perform Non-negative Matrix Factorization (NMF).

    Decompose a non-negative matrix X into an approximation of the product of two matrices H and W of smaller ranks.
    It is useful for dimension reduction, topic modeling, gene program extraction in Genomics, etc.

    The objective function is

        .. math::

            ||X - HW||_{beta} + alpha_H * l1_{ratio, H} * ||vec(H)||_1

            + 0.5 * alpha_H * (1 - l1_{ratio, H}) * ||H||_{Fro}^2

            + alpha_W * l1_{ratio, W} * ||vec(W)||_1

            + 0.5 * alpha_W * (1 - l1_{ratio, W}) * ||W||_{Fro}^2

    where

    :math:`||A||_{beta} = \\frac{1}{beta * (beta - 1)} \\sum_{i, j} X_{ij}^{beta} - beta * X_{ij} * Y_{ij}^{beta - 1} + (beta - 1) * Y_{ij}^{beta}` (Beta divergence)

    :math:`||vec(A)||_1 = \\sum_{i, j} abs(A_{ij})` (Element-wise L1 norm)

    :math:`||A||_{Fro}^2 = \\sum_{i, j} A_{ij}^2` (Frobenius norm)

    NMF uses the multiplicative update (MU) solver, either a batch version or an online version specified in ``update_method`` parameter, to minimize the objective function.

    Parameters
    ----------

    X: ``numpy.array`` or ``torch.tensor``
        The input non-negative matrix of shape (n_samples, n_features).
    n_components: ``int``
        Number of components.
    init: ``str``, optional, default: ``nndsvdar``
        Method for initialization on H and W matrices. Available options are: ``random``, ``nndsvd``, ``nndsvda``, ``nndsvdar``.
    beta_loss: ``str`` or ``float``
        Beta loss between the given matrix X and its approximation calculated by HW, which is used as the metric to be minimized during the computation.
        It can be a string from options:
            - ``frobenius``: L2 distance, same as ``beta_loss=2.0``.
            - ``kullback-leibler``:KL divergence, same as ``beta_loss=1.0``.
            - ``itakura-saito``: Itakura-Saito divergence, same as ``beta_loss=0``.
        Alternatively, it can also be a float number, which gives the beta parameter of the beta loss to be used.
    update_method: ``str``, optional, default: ``batch``
        Specify the updating method for H and W matrices.
        If ``batch``, NMF uses the multiplicative updating (MU) method.
        If ``online``, use the online MU method. This gives a faster convergence, and scales up to large matrices.
        Notice that ``online`` only works when ``beta=2.0``. For other beta loss, it switches back to ``batch`` method.
    max_iter: ``int``, optional, default: ``200``
        The maximum number of iterations to perform. This is used only when ``update_method="batch"``.
    tol: ``float``, optional, default: ``1e-4``
        The toleration used for convergence check.
    random_state: ``int``, optional, default: ``0``
        The random state used for reproducibility on the results.
    alpha_W: ``float``, optional, default: ``0.0``
        A numeric scale factor which multiplies the regularization terms related to W.
        If zero or negative, no regularization regarding W is considered.
    l1_ratio_W: ``float``, optional, default: ``0.0``
        The ratio of L1 penalty on W, must be between 0 and 1. And thus the ratio of L2 penalty on W is (1 - l1_ratio_W).
    alpha_H: ``float``, optional, default: ``0.0``
        A numeric scale factor which multiplies the regularization terms related to H.
        If zero or negative, no regularization regarding H is considered.
    l1_ratio_H: ``float``, optional, default: ``0.0``
        The ratio of L1 penalty on W, must be between 0 and 1. And thus the ratio of L2 penalty on H is (1 - l1_ratio_H).
    fp_precision: ``str``, optional, default: ``float``
        The numeric precision on the results.
        If ``float``, set precision to ``torch.float``; if ``double``, set precision to ``torch.double``.
        Alternatively, choose Pytorch's `torch dtype <https://pytorch.org/docs/stable/tensor_attributes.html>`_ of your own.
    online_max_pass: ``int``, optional, default: ``10``
        The maximum number of passes to perform. This is used only when ``update_method="online"``.
    online_chunk_size: ``int``, optional, default: ``2000``
        The chunk size when partitioning X regarding samples. This is used only when ``update_method="online"``.
    online_w_max_iter: ``int``, optional, default: ``200``
        The maximum number of iterations when updating W in Online algorithm. Used only when ``update_method="online"``.
    online_h_max_iter: ``int``, optinoal, default: ``50``
        The maximum number of iterations when updating H (or h, a chunk of H) in Online algorithm. Used only when ``update_method="online"``.
    use_gpu: ``bool``, optional, default: ``False``
        If ``True``, use GPU if available. Otherwise, use CPU only.

    Returns
    -------
    H: ``numpy.array``
        One of the resulting decomposed matrix of shape (n_samples, n_components). It represents the transformed coordinates of samples regarding components.
    W: ``numpy.array``
        The other resulting decomposed matrix of shape (n_components, n_features). It represents the composition of each component in terms of features.
    reconstruction_error: ``float``
        The Beta Loss between the origin matrix X and its approximation HW after NMF.

    Examples
    --------
    >>> H, W, err = run_nmf(X, n_components=20)

    >>> H, W, err = run_nmf(X, n_components=20, init='random', update_method='online')
    """
    if beta_loss == 'frobenius':
        beta_loss = 2
    elif beta_loss == 'kullback-leibler':
        beta_loss = 1
    elif beta_loss == 'itakura-saito':
        beta_loss = 0
    elif not (isinstance(beta_loss, int) or isinstance(beta_loss, float)):
        raise ValueError("beta_loss must be a valid value: either from ['frobenius', 'kullback-leibler', 'itakura-saito'], or a numeric value.")

    device_type = 'cpu'
    if use_gpu:
        if torch.cuda.is_available():
            device_type = 'cuda'
            print("Use GPU mode.")
        else:
            print("CUDA is not available on your machine. Use CPU mode instead.")

    if update_method in ['batch mu', 'online mu', 'batch hals', 'online hals']:
        if beta_loss != 2 and update_method == 'online':
            print("Cannot perform online update when beta not equal to 2. Switch to batch update method.")
            update_method = 'batch'

        if update_method == 'batch mu':
            model = NMFBatch(
                n_components=n_components,
                init=init,
                beta_loss=beta_loss,
                tol=tol,
                random_state=random_state,
                alpha_W=alpha_W,
                l1_ratio_W=l1_ratio_W,
                alpha_H=alpha_H,
                l1_ratio_H=l1_ratio_H,
                fp_precision=fp_precision,
                device_type=device_type,
                max_iter=max_iter,
            )
        elif update_method == 'online mu':
            model = NMFOnline(
                n_components=n_components,
                init=init,
                beta_loss=beta_loss,
                tol=tol,
                random_state=random_state,
                alpha_W=alpha_W,
                l1_ratio_W=l1_ratio_W,
                alpha_H=alpha_H,
                l1_ratio_H=l1_ratio_H,
                fp_precision=fp_precision,
                device_type=device_type,
                max_pass=online_max_pass,
                chunk_size=online_chunk_size,
                w_max_iter=online_w_max_iter,
                h_max_iter=online_h_max_iter,
            )
        elif update_method == 'batch hals':
            model = NMFBatchHALS(
                n_components=n_components,
                init=init,
                beta_loss=beta_loss,
                tol=tol,
                random_state=random_state,
                alpha_W=alpha_W,
                l1_ratio_W=l1_ratio_W,
                alpha_H=alpha_H,
                l1_ratio_H=l1_ratio_H,
                fp_precision=fp_precision,
                device_type=device_type,
                max_iter=max_iter,
            )
        else:
            model = NMFOnlineHALS(
                n_components=n_components,
                init=init,
                beta_loss=beta_loss,
                tol=tol,
                random_state=random_state,
                alpha_W=alpha_W,
                l1_ratio_W=l1_ratio_W,
                alpha_H=alpha_H,
                l1_ratio_H=l1_ratio_H,
                fp_precision=fp_precision,
                device_type=device_type,
                max_pass=online_max_pass,
                chunk_size=online_chunk_size,
                w_max_iter=online_w_max_iter,
                h_max_iter=online_h_max_iter,
            )

    else:
        raise ValueError("Parameter update_method must be a valid value from ['batch', 'online']!")

    H = model.fit_transform(X)
    W = model.W
    err = model.reconstruction_err

    if device_type == 'cpu':
        return H.numpy(), W.numpy(), err.numpy()
    else:
        return H.cpu().numpy(), W.cpu().numpy(), err.cpu().numpy()
