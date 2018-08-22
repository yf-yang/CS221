import numpy as np

def gen_f(w, a, b, lamb):
    '''
    input:
    -   a: (d, n)  np.ndarray
    -   b: (d, n)  np.ndarray
    output:
    -   f: a function with input w and lambda, which computes f(w) in O(d^2) time
    '''

    # preprocessing in O(nd^2) time    
    d, n = a.shape
    # (n, n) np.ndarray to restore middle output
    kernel = np.zeros((n, n))

    for i in range(d):
        for j in range(d):
            for k in range(n):
                kernel[i, j] += a[i, k] * a[j, k]

    for i in range(d):
        for j in range(d):
            for k in range(n):
                kernel[i, j] -= a[i, k] * b[j, k]

    for i in range(d):
        for j in range(d):
            for k in range(n):
                kernel[i, j] += b[i, k] * b[j, k]
    
    def f(w, lamb):
        '''
        input:
        -   w: (d,)    np.ndarray
        -   lamb: scaler    lambda
        output:
        -   out: scalar value of function f(w)
        '''

        # running in O(d^2) time
        out = 0

        for i in range(d):
            for j in range(d):
                out += kernel[i, j] * w[i] * w[j]

        for i in range(d):
            out += lamb * w[i] ** 2

        return out

    return f



