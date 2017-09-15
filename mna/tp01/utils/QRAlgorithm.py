from mna.tp01.utils.GramSchmidt import *
from copy import copy, deepcopy

class QRAlgorithm:

    @staticmethod
    def trivialEig (matrix, method=GramSchmidt.QR):
        n = matrix[0].size-1
        Q,R = method(matrix)
        H = matrix
        eig_val = Q.T.dot(H.dot(Q))
        eig_vec = Q
        lastValue = eig_val[1,1]+1
        for i in range(500):
        # while abs(eig_val[1,1]-lastValue) > 0:
            lastValue = eig_val[1,1]
            Q,R = method(eig_val)
            eig_val = R.dot(Q)
            eig_vec = eig_vec.dot(Q)

        return np.diagonal(eig_val), eig_vec

    @staticmethod
    def wilkinsonEig (matrix, method=GramSchmidt.QR):
        n = matrix[0].size
        Q,R = method(matrix)
        # To only found eigenValues
        # H = QRAlgorithm.HessenbergReduction(matrix)
        H = matrix
        eig_val = Q.T.dot(H.dot(Q))
        eig_vec = Q
        lastValue = eig_val[0,0]
        I = np.identity(n)
        while abs(eig_val[0,1]) > 0.0000000001:
            lastValue = eig_val[0,0]
            mu = QRAlgorithm.WilkinsonShift(eig_val[n-2,n-2], eig_val[n-1,n-1], eig_val[n-2,n-1])
            Q,R = method(eig_val-mu*I)
            eig_val = R.dot(Q) + I*mu
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
        H=np.array(matrix)
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
            H[k+2:n, k] = np.zeros((n - (k + 2)))


        return H