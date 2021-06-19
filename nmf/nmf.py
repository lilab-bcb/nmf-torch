import numpy as np
import torch
from typing import Union, Tuple

from nmf import NMFBatchMU, NMFBatchHALS, NMFBatchNnlsBpp, NMFOnlineMU, NMFOnlineHALS, NMFOnlineNnlsBpp


def run_nmf(
    X: Union[np.array, torch.tensor],
    n_components: int,
    init: str = "nndsvdar",
    beta_loss: Union[str, float] = "frobenius",
    algo: str = "hals",
    mode: str = "batch",
    tol: float = 1e-4,
    random_state: int = 0,
    use_gpu: bool = False,
    alpha_W: float = 0.0,
    l1_ratio_W: float = 0.0,
    alpha_H: float = 0.0,
    l1_ratio_H: float = 0.0,
    fp_precision: Union[str, torch.dtype] = "float",
    batch_max_iter: int = 500,
    batch_hals_tol: float = 0.05,
    batch_hals_max_iter: int = 200,
    online_max_pass: int = 20,
    online_chunk_size: int = 5000,
    online_chunk_max_iter: int = 200,
    online_h_tol: float = 0.05,
    online_w_tol: float = 0.05,    
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
    algo: ``str``, optional, default: ``hals``
        Choose from ``mu`` (Multiplicative Update), ``hals`` (Hierarchical Alternative Least Square) and ``bpp`` (alternative non-negative least squares with Block Principal Pivoting method).
    mode: ``str``, optional, default: ``batch``
        Learning mode. Choose from ``batch`` and ``online``. Notice that ``online`` only works when ``beta=2.0``. For other beta loss, it switches back to ``batch`` method.        
    tol: ``float``, optional, default: ``1e-4``
        The toleration used for convergence check.
    random_state: ``int``, optional, default: ``0``
        The random state used for reproducibility on the results.
    use_gpu: ``bool``, optional, default: ``False``
        If ``True``, use GPU if available. Otherwise, use CPU only.
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
    batch_max_iter: ``int``, optional, default: ``500``
        The maximum number of iterations to perform for batch learning.
    batch_hals_tol: ``float``, optional, default: ``0.05``
        For HALS, we have the option of using HALS to mimic BPP for a possible better loss. The mimic works as follows: update H by HALS several iterations until the maximal relative change < batch_hals_tol. Then update W similarly.
    batch_hals_max_iter: ``int``, optional, default: ``200``
        Maximal iterations of updating H & W for mimic BPP. If this parameter set to 1, it is the standard HALS. 
    online_max_pass: ``int``, optional, default: ``20``
        The maximum number of online passes of all data to perform.
    online_chunk_size: ``int``, optional, default: ``5000``
        The chunk / mini-batch size for online learning.
    online_chunk_max_iter: ``int``, optional, default: ``200``
        The maximum number of iterations for updating H or W in online learning.
    online_h_tol: ``float``, optional, default: 0.05
        The tolerance for updating H in each chunk in online learning.
    online_w_tol: ``float``, optional, default: 0.05
        The tolerance for updating W in each chunk in online learning.

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
    >>> H, W, err = run_nmf(X, n_components=20, init='random', algo='mu', mode='online')
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

    if algo not in {'mu', 'hals', 'bpp'}:
        raise ValueError("Parameter algo must be a valid value from ['mu', 'hals', 'bpp']!")
    if mode not in {'batch', 'online'}:
        raise ValueError("Parameter mode must be a valid value from ['batch', 'online']!")
    if beta_loss != 2 and mode == 'online':
        print("Cannot perform online update when beta not equal to 2. Switch to batch update method.")
        mode = 'batch'

    model_class = None
    kwargs = {'alpha_W': alpha_W, 'l1_ratio_W': l1_ratio_W, 'alpha_H': alpha_H, 'l1_ratio_H': l1_ratio_H, 'fp_precision': fp_precision, 'device_type': device_type}

    if mode == 'batch':
        kwargs['max_iter'] = batch_max_iter
        if algo == 'mu':
            model_class = NMFBatchMU
        elif algo == 'hals':
            model_class = NMFBatchHALS
            kwargs['hals_tol'] = batch_hals_tol
            kwargs['hals_max_iter'] = batch_hals_max_iter
        else:
            model_class = NMFBatchNnlsBpp
    else:
        kwargs['max_pass'] = online_max_pass
        kwargs['chunk_size'] = online_chunk_size
        if algo == 'mu' or algo == 'hals':
            kwargs['chunk_max_iter'] = online_chunk_max_iter
            kwargs['h_tol'] = online_h_tol
            kwargs['w_tol'] = online_w_tol
            model_class = NMFOnlineMU if algo == 'mu' else NMFOnlineHALS
        else:
            model_class = NMFOnlineNnlsBpp

    model = model_class(
                n_components=n_components,
                init=init,
                beta_loss=beta_loss,
                tol=tol,
                random_state=random_state,
                **kwargs            
            )

    H = model.fit_transform(X)
    W = model.W
    err = model.reconstruction_err

    return H.cpu().numpy(), W.cpu().numpy(), err.cpu().numpy()
