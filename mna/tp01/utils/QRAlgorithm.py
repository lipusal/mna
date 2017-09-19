import numpy as np
from numpy.linalg import norm
from copy import copy
from mna.tp01.utils.Utils import *

class QRAlgorithm:

    @staticmethod
    def GramSchmidt(matrix):
        n = len(matrix)
        q = np.zeros((n, n))
        r = np.zeros((n, n))
        for i in range(n):
            u = col(matrix[:, i])
            aux = u
            for j in range(i):
                aux = aux - (col(q[:, j]) * inner(u, (q[:, j])))
            u = aux
            e = u / np.linalg.norm(u)

            for j in range(i, n):
                r[i, j] = inner(matrix[:, j], e)

            for j in range(n):
                q[j, i] = e[j]

        return q, r

    @staticmethod
    def __signZero(n):
        if n == 0:
            return 1
        else:
            return np.sign(n)

    @staticmethod
    def HouseHolder(matrix):
        R = copy(np.asmatrix(matrix).astype(float))
        m = len(matrix)
        Q = np.eye(m).astype(float)
        for k in range(m - 1):
            u = copy(R[k:m, k])
            u[0] = u[0] + QRAlgorithm.__signZero(u[0]) * norm(u)
            u = u / norm(u)
            R[k:m, k:m] = R[k:m, k:m] - (2.0 * u) * (u.transpose() * R[k:m, k:m])
            Q[0:m, k:m] = Q[0:m, k:m] - (Q[0:m, k:m] * (2.0 * u)) * u.transpose()

        for i in range(m - 1):
            for j in range(i + 1):
                R[i + 1, j] = 0

        Q = np.asarray(Q)
        R = np.asarray(R)

        return Q, R

    @staticmethod
    def reflector(vector):
        e1 = np.zeros(np.size(vector))
        e1[0] = 1
        u = vector - np.dot(np.linalg.norm(vector), e1)
        u = u / np.linalg.norm(u)

        return np.identity(np.size(vector)) - np.dot(2, np.dot(col(u),row(u)))
