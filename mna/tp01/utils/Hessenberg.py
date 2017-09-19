from mna.tp01.utils.GramSchmidt import *
from mna.tp01.utils.HouseHolder import *


class Hessenberg:

    @staticmethod
    def HessenbergReductionWithReflector(matrix):
        # Reduce A to a Hessenberg matrix H so that A and H are similar:
        H = np.array(matrix)
        n = H[0].size
        U = np.identity(n)
        for k in range(n - 2):
            x = copy(col(H[k + 1:n, k]))
            v = HouseHolder.reflector(row(x)[0])
            identity = np.identity(matrix.shape[0] - v.shape[0])
            z1 = np.zeros([identity.shape[0], v.shape[1]])
            z2 = np.zeros([v.shape[0], identity.shape[1]])
            p1 = np.concatenate((identity, z1), axis=1)
            p2 = np.concatenate((z2, v), axis=1)
            p = np.concatenate([p1, p2])
            H = p.dot(H.dot(np.linalg.inv(p)))
            U = U.dot(p)

        return H, U

    @staticmethod
    def HessenbergReduction(matrix):
        # Reduce A to a Hessenberg matrix H so that A and H are similar:
        H = np.array(matrix)
        n = H[0].size
        for k in range(n - 2):
            x = copy(col(H[k + 1:n, k]))
            e1 = col(np.zeros(n - (k + 1)))
            e1[0] = 1
            sgn = np.sign(x[0])

            v = (x + e1 * sgn * np.linalg.norm(x))
            v = v / np.linalg.norm(v)

            H[k + 1:n, k:n] = H[k + 1:n, k:n] - 2 * v * (v.T.dot(H[k + 1:n, k:n]))
            H[:, k + 1:n] = H[:, k + 1:n] - 2 * (H[:, k + 1:n].dot(v)).dot(v.T)
            H[k + 2:n, k] = np.zeros((n - (k + 2)))

        return H
