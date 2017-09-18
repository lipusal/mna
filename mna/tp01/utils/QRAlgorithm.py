from mna.tp01.utils.GramSchmidt import *
import sys
from copy import copy, deepcopy
from mna.tp01.utils.HouseHolder import *


__precision__ = sys.float_info.epsilon

class QRAlgorithm:

    # This is the most basic algorithm
    # checks when the subdiagonal converges to 0
    # Calculate the eigenVectors multiplying every iteration
    @staticmethod
    def trivialEig (matrix, method=GramSchmidt.QR):
        n = matrix[0].size-1
        Q,R = method(matrix)
        H = matrix
        eig_val = Q.T.dot(H.dot(Q))
        eig_vec = Q
        while sum(abs(np.diag(eig_val[1:n,:]))> __precision__*n):
            Q,R = method(eig_val)
            eig_val = R.dot(Q)
            eig_vec = eig_vec.dot(Q)

        return np.diagonal(eig_val), eig_vec

    # This is the most basic algorithm
    # checks in order each element of subdiagonal is 0
    # Calculate the eigenVectors multiplying every iteration
    @staticmethod
    def betterCheckEig (matrix, method=GramSchmidt.QR):
        n = matrix[0].size-1
        Q,R = method(matrix)
        H = matrix
        eig_val = Q.T.dot(H.dot(Q))
        eig_vec = Q
        known = 0
        while known<n:
            while n-1-known >= 0 and eig_val[n-known, n-1-known] < sys.float_info.epsilon:
                known += 1

            Q,R = method(eig_val)
            eig_val = R.dot(Q)
            eig_vec = eig_vec.dot(Q)

        return np.diagonal(eig_val), eig_vec

    @staticmethod
    def wilkinsonEig (matrix, method=GramSchmidt.QR):
        n = matrix[0].size
        Q,R = method(matrix)
        H = matrix
        eig_val = Q.T.dot(H.dot(Q))
        eig_vec = Q
        I = np.identity(n)
        while sum(abs(np.diag(eig_val[1:n,:]))>__precision__):
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

    @staticmethod
    def HessenbergReductionWithReflector(matrix):
        # Reduce A to a Hessenberg matrix H so that A and H are similar:
        H = np.array(matrix)
        n = H[0].size
        U = np.identity(n)
        for k in range(n - 2):
            x = deepcopy(col(H[k + 1:n, k]))
            v = HouseHolder.reflector(row(x)[0])
            identity = np.identity(matrix.shape[0] - v.shape[0])
            z1 = np.zeros([identity.shape[0], v.shape[1]])
            z2 = np.zeros([ v.shape[0], identity.shape[1]])
            p1 = np.concatenate((identity, z1), axis=1)
            p2 = np.concatenate((z2, v), axis=1)
            p = np.concatenate([p1,p2])
            H = p.dot(H.dot(np.linalg.inv(p)))
            U = U.dot(p)

        return H, U



    @staticmethod
    def FrancisAlgorithm(vector):
        pass
