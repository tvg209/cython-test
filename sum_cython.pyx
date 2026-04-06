
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
# distutils: extra_compile_args = -O3 -march=native -fopenmp
# distutils: extra_link_args = -fopenmp

from cython.parallel cimport prange

def sum_cython(double[:] a):
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double s = 0.0
    n = a.shape[0]

    for i in prange(n, nogil=True, schedule='static'):
        s += a[i]

    return s
