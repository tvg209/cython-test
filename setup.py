

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        "banded_solver.pyx",
        language_level="3",
        annotate=True
    ),
    include_dirs=[numpy.get_include()],
)
