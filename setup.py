
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension(
    name="banded_solver",
    sources=["banded_solver.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=["/O2", "/openmp", "/fp:fast"],
    extra_link_args=["/openmp"],
)

setup(
    ext_modules=cythonize(ext, language_level="3"),
)
