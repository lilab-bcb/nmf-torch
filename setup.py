from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from pathlib import Path
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

extensions = [
    Extension("nmf.cylib.nnls_bpp_utils", ["ext_modules/nnls_bpp_utils.pyx"]),
]

setup(
    name="nmf-torch",
    use_scm_version=True,
    description="A PyTorch implementation on Non-negative Matrix Factorization.",
    long_description=long_description,
    url="https://github.com/lilab-bcb/nmf-torch",
    author="Yiming Yang, Bo Li",
    author_email="cumulus-support@googlegroups.com",
    classifiers=[  # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: POSIX :: Linux",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="matrix factorization",
    packages=find_packages(),
    install_requires=[
        l.strip() for l in Path("requirements.txt").read_text("utf-8").splitlines()
    ],
    ext_modules=cythonize(extensions),
    setup_requires=["Cython", "setuptools_scm"],
    python_requires="~=3.6",
)
