
# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: language_level=3

import numpy as np
cimport numpy as np

def mv_banded_cython(double[:, :] Ab, int k, double[:] x):
    cdef Py_ssize_t n = x.shape[0]
    cdef double[:] y = np.zeros(n, dtype=np.float64)
    cdef Py_ssize_t i, j

    for i in range(n):
        y[i] = Ab[k, i] * x[i]
        for j in range(1, k+1):
            if i - j >= 0:
                y[i] += Ab[k-j, i] * x[i-j]
            if i + j < n:
                y[i] += Ab[k+j, i] * x[i+j]
    return y


def cholesky_banded_cython(double[:, :] Ab, int k):
    cdef Py_ssize_t n = Ab.shape[1]
    cdef Py_ssize_t i, j, d, jm, im
    cdef double s, sumjj
    cdef int jmax

    for j in range(n):

        sumjj = 0.0
        for d in range(1, k+1):
            i = j - d
            if i < 0:
                break
            sumjj += Ab[k + (i - j), j] ** 2

        Ab[k, j] = (<double> np.sqrt(Ab[k, j] - sumjj))

        jmax = min(n - 1, j + k)
        for i in range(j+1, jmax+1):
            s = 0.0
            for d in range(1, k+1):
                jm = j - d
                im = i - d
                if jm < 0 or im < 0:
                    break
                if abs(jm - i) <= k:
                    s += Ab[k + (jm - i), i] * Ab[k + (jm - j), j]

            Ab[k + (j - i), i] = (Ab[k + (j - i), i] - s) / Ab[k, j]

    return Ab


def solve_cholesky_banded_cython(double[:, :] Ab, int k, double[:] b):
    cdef Py_ssize_t n = b.shape[0]
    cdef double[:] y = np.copy(b)
    cdef double[:] x
    cdef Py_ssize_t i, j, d

    for i in range(n):
        for d in range(1, k+1):
            j = i - d
            if j < 0:
                break
            y[i] -= Ab[k + (j - i), i] * y[j]
        y[i] /= Ab[k, i]

    x = np.copy(y)

    for i in range(n-1, -1, -1):
        for d in range(1, k+1):
            j = i + d
            if j >= n:
                break
            x[i] -= Ab[k + (i - j), j] * x[j]
        x[i] /= Ab[k, i]

    return x
