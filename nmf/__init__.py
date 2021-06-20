
#from ._nmf_batch_mu import NMFBatchMU
#from ._nmf_batch_hals import NMFBatchHALS
#from ._nmf_batch_nnls_bpp import NMFBatchNnlsBpp
#from ._nmf_online_mu import NMFOnlineMU
#from ._nmf_online_hals import NMFOnlineHALS
#from ._nmf_online_nnls_bpp import NMFOnlineNnlsBpp

from .nmf import run_nmf, integrative_nmf

#from ._inmf_batch_mu import INMFBatchMU
#from ._inmf_batch_hals import INMFBatchHALS
#from ._inmf_batch_nnls_bpp import INMFBatchNnlsBpp
#from ._inmf_online_mu import INMFOnlineMU
#from ._inmf_online_hals import INMFOnlineHALS
#from ._inmf_online_nnls_bpp import INMFOnlineNnlsBpp

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # < Python 3.8: Use backport module
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version('nmf-torch')
    del version
except PackageNotFoundError:
    pass
