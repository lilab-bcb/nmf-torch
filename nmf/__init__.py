from .nmf import run_nmf
from ._nmf_batch import NMFBatch
from ._nmf_batch_hals import NMFBatchHALS
from ._nmf_online import NMFOnline
from ._nmf_online_hals import NMFOnlineHALS
from ._inmf_batch import INMFBatch
from ._inmf_batch_hals import INMFBatchHALS
from ._inmf_batch_hals_wrong import INMFBatchHALSWrong
from ._inmf_batch_nnls_bpp import INMFBatchNnlsBpp
from ._inmf_online import INMFOnline
from ._inmf_batch_nnls_bpp import INMFBatchNnlsBpp

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # < Python 3.8: Use backport module
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version('nmf-torch')
    del version
except PackageNotFoundError:
    pass
