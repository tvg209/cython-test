
def sum_cython(double[:] a):
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double s = 0
    for i in range(n):
        s += a[i]
    return s
