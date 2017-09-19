import numpy as np

class InverseIteration:


    #receives a matrix and an eigenvalue and returns an eigenvector associated with the given eigenvalue
    @staticmethod
    def gettingEigenVector(matrix, eigenvalue):
        n = len(matrix)
        eye = np.eye(n)
        eye = eye*eigenvalue
        # eigenvector = np.random.rand(1,n)
        eigenvector = np.ones((1,n)).T
        inv = np.linalg.inv(matrix - eye)
        aux = np.dot(inv, eigenvector)
        aux = aux/np.linalg.norm(aux)
        i = 0
        while np.linalg.norm(np.subtract(eigenvector,aux))>10**-5 and i<100:
            i = i + 1
            eigenvector = aux
            aux = np.dot(inv, eigenvector)
            aux = aux/np.linalg.norm(aux)

        return eigenvector