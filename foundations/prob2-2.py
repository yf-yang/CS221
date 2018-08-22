import numpy as np

def compute_minimum_cost(c):
    '''
    input:
    -   c: (n, n) np.ndarray, cost matrix (cost function) of each location
    output:
    -   min_cost: scalar, minimum cost of all the paths
    '''

    h, w = c.shape
    assert h == w
    n = h
    min_cost = 100000000

    # temp memory to restore cost to current location
    path_cost = np.zeros((n, n))
    for i in range(n):
        path_cost[i, 0] = c[i, 0]
    for j in range(1, n):
        path_cost[0, j] = c[0, j]
    for i in range(1, n):
        for j in range(1, n):
            path_cost[i, j] = c[i, j] + min(path_cost[i-1, j], path_cost[i, j-1])
    return path_cost[n-1, n-1]