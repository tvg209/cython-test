
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension(
    name="banded_solver",
    sources=["banded_solver.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-O3", "-march=native", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)

setup(
    ext_modules=cythonize(ext, language_level="3", annotate=False),
)
