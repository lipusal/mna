import numpy as np
from mna.tp01.utils.Utils import *

class GrandSchmidt:

    @staticmethod
    def QR (matrix):
        n = len(matrix)
        q = np.zeros((n, n))
        r = np.zeros((n, n))
        for i in range(n):
            u = col(matrix[:, i])
            aux = u
            for j in range(i):
                aux = aux - ( col(q[:,j]) * inner(u,(q[:,j])) )
            u = aux
            e = u / np.linalg.norm(u)

            for j in range(i, n):
                r[i,j] = inner(matrix[:,j],e)

            for j in range(n):
                q[j,i] = e[j]
        return q, r

