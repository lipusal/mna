import numpy as np
from numpy.linalg import norm
from copy import copy, deepcopy


def signZero(n):
    if n == 0:
        return 1
    else:
        return np.sign(n)

class HouseHolder:
    def qr(matrix):
        R = deepcopy(np.asmatrix(matrix).astype(float))
        m = len(matrix)
        Q = np.eye(m).astype(float)
        for k in range(m - 1):
            u = deepcopy(R[k:m, k])
            u[0] = u[0] + signZero(u[0]) * norm(u)
            u = u / norm(u)
            R[k:m, k:m] = R[k:m, k:m] - (2.0 * u) * (u.transpose() * R[k:m, k:m])
            Q[0:m, k:m] = Q[0:m, k:m] - (Q[0:m, k:m] * (2.0 * u)) * u.transpose()

        for i in range(m - 1):
            for j in range(i + 1):
                R[i + 1, j] = 0

        Q = np.asarray(Q)
        R = np.asarray(R)

        return Q, R
