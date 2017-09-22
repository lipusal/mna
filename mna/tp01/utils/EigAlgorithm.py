from mna.tp01.utils.Hessenberg import Hessenberg
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
        eigenvector = np.ones((1, n)).T
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
    def wilkinsonEigNoHess (matrix, method=QRAlgorithm.GramSchmidt):
        n = matrix.shape[0]
        orig = copy(matrix)

        eig_val = np.zeros(n)
        print("Calculating eigenValues")
        EigAlgorithm.rec3Wilkinson(matrix, method, eig_val)

        eig_vec = np.zeros((n,n))
        print("Calculating eigenVectors")
        for i in range(len(eig_val)):
            aux = EigAlgorithm.InverseIteration(matrix, eig_val[i])
            eig_vec[:, i] = aux.reshape(n)

        return eig_val, eig_vec

    @staticmethod
    def wilkinsonEig (matrix, method=QRAlgorithm.GramSchmidt):
        n = matrix.shape[0]
        orig = copy(matrix)
        print("Reducing to hessenberg")
        H, Q = Hessenberg.HessenbergReductionWithReflector(matrix)

        eig_val = np.zeros(n)

        print("Calculating eigenValues")
        EigAlgorithm.rec3Wilkinson(H, method, eig_val)

        eig_vec = np.zeros((n,n))
        print("Calculating eigenVectors")
        for i in range(len(eig_val)):
                aux = EigAlgorithm.InverseIteration(H, eig_val[i])
                eig_vec[:, i] = aux.reshape(n)

        return eig_val, eig_vec.dot(Q)

    @staticmethod
    def recWilkinson(matrix, method, answer):
        A = matrix
        n = A[0].size
        if n==1:
            answer[0] = A[0][0]
        else:
            I = np.identity(n)
            lastVal = A[n-1,n-1] + 1
            while abs(lastVal - A[n-1,n-1]) > __precision__  :
                mu = EigAlgorithm.wilkinsonShift(A[n-2,n-2], A[n-1,n-1], A[n-2,n-1])
                Q,R = method(A-mu*I)
                A = R.dot(Q) + I*mu
                lastVal = A[n-1,n-1]

            answer[n-1] = A[n-1,n-1]
            EigAlgorithm.recWilkinson(A[0:n-1, 0:n-1], method, answer)

    @staticmethod
    def rec2Wilkinson(matrix, method, answer):
        A = matrix
        n = np.shape(A)[0]
        if n==1:
            answer[0] = A[0][0]
        else:

            mid = n//2
            i=1
            while i<n:
                search = int(mid - (i//2 * (-1 + 2*(i%2))))
                # search = i
                if abs(A[search, search-1])<0.1:
                    if search==n-1 :
                        answer[search] = A[search,search]
                        EigAlgorithm.rec2Wilkinson(A[0:search, 0:search], method, answer)
                        return

                    if search == 1:
                        answer[search] = A[search,search]
                        answer[0] = A[0,0]
                        EigAlgorithm.rec2Wilkinson(A[2:n, 2:n], method, answer[2:n])
                        return

                    EigAlgorithm.rec2Wilkinson(A[0:search, 0:search],method, answer[0:search])
                    EigAlgorithm.rec2Wilkinson(A[search:n, search:n],method, answer[search:n])
                    return
                i+=1


            lastVal = A[n-1,n-1] + 1
            I = np.identity(n)
            while abs(lastVal - A[n-1,n-1]) > __precision__ :
                mu = EigAlgorithm.wilkinsonShift(A[n-2,n-2], A[n-1,n-1], A[n-2,n-1])
                Q,R = method(A-mu*I)
                A = R.dot(Q) + I*mu
                lastVal = A[n-1,n-1]

            answer[n-1] = A[n-1,n-1]
            EigAlgorithm.rec2Wilkinson(A[0:n-1, 0:n-1], method, answer)


    @staticmethod
    def rec3Wilkinson(matrix, method, answer):
        A = matrix
        n = np.shape(A)[0]
        if n==0:
            return
        if n==1:
            answer[0] = A[0][0]
        else:

            lastVal = A[n-1,n-1] + 1
            I = np.identity(n)
            while abs(lastVal - A[n-1,n-1]) > __precision__ :
                mu = EigAlgorithm.wilkinsonShift(A[n-2,n-2], A[n-1,n-1], A[n-2,n-1])
                Q,R = method(A-mu*I)
                A = R.dot(Q) + I*mu
                mid = n//2
                i=0
                while i<n:
                    search = int(mid - (i//2 * (-1 + 2*(i%2))))
                    # search = i
                    if abs(lastVal - A[search,search]) < 0.1 :
                        if search==n-1 :
                            answer[search] = A[search,search]
                            EigAlgorithm.rec3Wilkinson(A[0:search, 0:search], method, answer)
                            return

                        if search == 1:
                            answer[search] = A[search,search]
                            answer[0] = A[0,0]
                            EigAlgorithm.rec3Wilkinson(A[2:n, 2:n], method, answer[2:n])
                            return

                        EigAlgorithm.rec3Wilkinson(A[0:search, 0:search],method, answer[0:search])
                        EigAlgorithm.rec3Wilkinson(A[search:n, search:n],method, answer[search:n])
                        return
                    i+=1
                lastVal = A[n-1,n-1]

            answer[n-1] = A[n-1,n-1]
            EigAlgorithm.rec3Wilkinson(A[0:n-1, 0:n-1], method, answer)


    @staticmethod
    def divide(matrix, method, answer):
        n = matrix.shape[0]
        if n== 0:
            return
        if n == 1:
            answer[0] = matrix[0][0]
            return
        mid = n//2
        i=1
        while i<n:
            search = int(mid - (i/2 * (-1 + 2*(i%2))))
            # print(abs(matrix[search, search-1]))
            if abs(matrix[search, search-1])<__precision__:
                if search==n-1 :
                    answer[search] = matrix[search,search]
                    EigAlgorithm.rec2Wilkinson(A[0:search, 0:search], method, answer)
                    return

                if search == 1:
                    answer[search] = matrix[search,search]
                    answer[0] = matrix[0,0]
                    return

                EigAlgorithm.rec2Wilkinson(matrix[0:search, 0:search],method, answer[0:search])
                EigAlgorithm.rec2Wilkinson(matrix[search:n, search:n],method, answer[search:n])
            i+=1

        QRAlgorithm.recWilkinson(matrix,method, answer)

    # Calculate Wilkinson's shift for symmetric matrices:
    @staticmethod
    def wilkinsonShift(a, b, c):
        delta = (a-c) / 2
        return c - np.sign(delta) * b**2 / (abs(delta) + np.sqrt(delta**2 + b**2))

    @staticmethod
    def francisAlgorithm(vector):
        pass
