from mna.tp01.utils.GrandSchmidt import *
from copy import copy, deepcopy

class QRAlgorithm:

    @staticmethod
    def QR (matrix, method=GrandSchmidt.QR):
        n = matrix[0].size
        Q,R = method(matrix)
        eig_val = Q.T.dot(matrix.dot(Q))
        eig_vec = Q
        lastValue = matrix[0,0]
        I = np.identity(n)
        # while abs(eig_val[0,0]-lastValue) > 0.01:
        for i in range(50):
            lastValue = eig_val[0,0]
            print(lastValue)
            mu = QRAlgorithm.WilkinsonShift(eig_val[n-2,n-2], eig_val[n-1,n-1], eig_val[n-2,n-1])
            # This line should use faster Hessenberg reduction:
            Q,R = method(eig_val-mu*I)
            # This line needs speeding up, currently O(n^3) operations!:
            eig_val = R*Q + mu*I
            # Q,R = method(matrix)
            # eig_val = Q.T.dot(eig_val.dot(Q))
            eig_vec = eig_vec.dot(Q)

        return np.diagonal(eig_val), eig_vec

    @staticmethod
    def WilkinsonShift( a, b, c ):
        # Calculate Wilkinson's shift for symmetric matrices:
        delta = (a-c)/2
        return c - np.sign(delta)*b**2/(abs(delta) + np.sqrt(delta**2+b**2))

    @staticmethod
    def HessenbergReduction(matrix):
    # Reduce A to a Hessenberg matrix H so that A and H are similar:
        H=deepcopy(matrix)
        n = H[0].size
        for k in range(n-2):
            x = deepcopy(col(H[k + 1:n, k]))
            e1 = col(np.zeros(n-(k+1)))
            e1[0] = 1
            sgn = np.sign(x[0])

            v = (x + e1*sgn*np.linalg.norm(x))
            v = v/np.linalg.norm(v)

            H[k+1:n, k:n] = H[k+1:n, k:n] - 2 * v * (v.T.dot(H[k+1:n, k:n]))
            H[:, k+1:n] = H[:, k+1:n] - 2 * (H[:, k+1:n].dot(v)).dot(v.T)
            H[k+2:n, k] = col(np.zeros((n - (k + 2))))


        return H