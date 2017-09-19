from mna.tp01.utils.QRAlgorithm import *
import sys
import time

__precision__ = 10**-5
# __precision__ = sys.float_info.epsilon


class EigAlgorithm:

    # Receives a matrix and an eigenvalue and returns an eigenvector associated with the given eigenvalue
    @staticmethod
    def InverseIteration(matrix, eigenvalue):
        n = len(matrix)
        eye = np.eye(n)
        eye = eye*eigenvalue
        # eigenvector = np.random.rand(1,n)
        eigenvector = np.ones((1,n)).T
        inv = np.linalg.inv(matrix - eye)
        aux = np.dot(inv, eigenvector)
        aux = aux/np.linalg.norm(aux)
        i = 0
        while np.linalg.norm(np.subtract(eigenvector,aux)) > 10**-5 and i < 100:
            i = i + 1
            eigenvector = aux
            aux = np.dot(inv, eigenvector)
            aux = aux/np.linalg.norm(aux)

        return eigenvector

# This is the most basic algorithm
    # checks when the subdiagonal converges to 0
    # Calculate the eigenVectors multiplying every iteration
    @staticmethod
    def trivialEig (matrix, method=QRAlgorithm.GramSchmidt):
        n = matrix[0].size-1
        Q,R = method(matrix)
        H = matrix
        eig_val = Q.T.dot(H.dot(Q))
        eig_vec = Q
        while sum(abs(np.diag(eig_val[1:n,:])) > __precision__ * n):
            Q, R = method(eig_val)
            eig_val = R.dot(Q)
            eig_vec = eig_vec.dot(Q)

        return np.diagonal(eig_val), eig_vec

    # This is the most basic algorithm
    # checks in order each element of subdiagonal is 0
    # Calculate the eigenVectors multiplying every iteration
    @staticmethod
    def betterCheckEig (matrix, method=QRAlgorithm.GramSchmidt):
        n = matrix[0].size-1
        Q,R = method(matrix)
        H = matrix
        eig_val = Q.T.dot(H.dot(Q))
        eig_vec = Q
        known = 0
        while known < n:
            while n-1-known >= 0 and eig_val[n-known, n-1-known] < sys.float_info.epsilon:
                known += 1

            Q, R = method(eig_val)
            eig_val = R.dot(Q)
            eig_vec = eig_vec.dot(Q)

        return np.diagonal(eig_val), eig_vec

    @staticmethod
    def wilkinsonEig2 (matrix, method=QRAlgorithm.GramSchmidt):
        n = matrix[0].size
        Q, R = method(matrix)
        H = matrix
        eig_val = Q.T.dot(H.dot(Q))
        eig_vec = Q
        I = np.identity(n)
        known = 1
        while known<n:
            while n-1-known >= 0 and eig_val[n-known, n-1-known] < __precision__:
                known += 1
                print(known)

            mu = EigAlgorithm.wilkinsonShift(eig_val[n-known-2, n-known-2], eig_val[n-known-1, n-known-1], eig_val[n-known-2, n-known-1])
            Q,R = method(eig_val-mu*I)
            eig_val = R.dot(Q) + I*mu
            eig_vec = eig_vec.dot(Q)

        return np.diagonal(eig_val), eig_vec

    @staticmethod
    def wilkinsonEig (matrix, method=QRAlgorithm.GramSchmidt):
        n = matrix.shape[0]
        orig = copy(matrix)
        t0 = time.time()
        print("RED IT TOOK " + str(time.time()-t0))
        eig_val = np.zeros(n)
        print("Calculating eigVal")
        t0= time.time()
        EigAlgorithm.rec2Wilkinson(matrix, method, eig_val)
        print("IT took " + str(time.time()-t0))
        # eig_vec = InverseIteration.gettingEigenVector(matrix, eig_val)
        eig_vec = np.zeros((n,n))
        print("Calculating eigVec")
        t0= time.time()
        for i in range(len(eig_val)):
                # print("EIGVECT " + str(i))
                aux = EigAlgorithm.InverseIteration(matrix, eig_val[i])
                eig_vec[:, i] = aux.reshape(n)

        print("IT took " + str(time.time()-t0))

        return eig_val, eig_vec

    @staticmethod
    def recWilkinson(matrix, method, answer):
        A = matrix
        n = A[0].size
        print(n)
        if n==1:
            answer[0] = A[0][0]
        else:
            I = np.identity(n)
            while abs(A[n-1,n-2]) > __precision__ :
                mu = EigAlgorithm.wilkinsonShift(A[n-2,n-2], A[n-1,n-1], A[n-2,n-1])
                Q,R = method(A-mu*I)
                A = R.dot(Q) + I*mu

            answer[n-1] = A[n-1,n-1]
            EigAlgorithm.recWilkinson(A[0:n-1, 0:n-1], method, answer)

    @staticmethod
    def rec2Wilkinson(matrix, method, answer):
        A = matrix
        n = A[0].size
        # print("---------------------" + str(n))
        if n==1:
            answer[0] = A[0][0]
        else:
            lastVal = A[n-1,n-1] + 1
            # QRAlgorithm.HessenbergReductionWithReflector(matrix)

            I = np.identity(n)
            while abs(lastVal - A[n-1,n-1]) > __precision__ :
                t0 = time.time()
                # print("Shifting")
                mu = EigAlgorithm.wilkinsonShift(A[n-2,n-2], A[n-1,n-1], A[n-2,n-1])
                # print(time.time() - t0)
                # t0 = time.time()
                # print("Calculating first QR")
                Q,R = method(A-mu*I)
                # print(time.time() - t0)
                # t0 = time.time()
                # print("Calculating next A")
                A = R.dot(Q) + I*mu
                # print(time.time() - t0)
                # t0 = time.time()
                lastVal = A[n-1,n-1]

            answer[n-1] = A[n-1,n-1]
            EigAlgorithm.rec2Wilkinson(A[0:n-1, 0:n-1], method, answer)

    # Calculate Wilkinson's shift for symmetric matrices:
    @staticmethod
    def wilkinsonShift(a, b, c):
        delta = (a-c) / 2
        return c - np.sign(delta) * b**2 / (abs(delta) + np.sqrt(delta**2 + b**2))

    @staticmethod
    def francisAlgorithm(vector):
        pass
