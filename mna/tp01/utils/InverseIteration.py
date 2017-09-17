import numpy as np

class InverseIteration:


    #receives a matrix and an eigenvalue and returns an eigenvector associated with the given eigenvalue
    @staticmethod
    def gettingEigenVector(matrix, eigenvalue):
        n = len(matrix)
        eye = np.eye(n)
        eye = eye*eigenvalue
        eigenvector = np.random.rand(1,n)
        eigenvector = eigenvector.T
        aux = np.dot(np.linalg.inv(matrix - eye), eigenvector)
        aux = aux/np.linalg.norm(aux)
        i = 0
        while np.linalg.norm(np.subtract(eigenvector,aux))>0.0000005 and i<100:
        #while (eigenvector==aux).all() == False:
        #for k in range(20):
            i = i + 1
            eigenvector = aux
            aux = np.dot(np.linalg.inv(matrix - eye), eigenvector)
            aux = aux/np.linalg.norm(aux)

        #print (eigenvector)
        return eigenvector


matrix = np.matrix('1 1 2; 3 4 5; 4 4 5')

#matrix = np.matrix('3650000 2675000 7300000; 2675000 1968500 5350000; 7300000 5350000 14600000')
eigenvalues,eigenvectors = np.linalg.eig(matrix)
#print(eigenvectors)
#matrix = np.dot(matrix.T,matrix)
#print(matrix)
#104.4879  9.5197   35.9204
InverseIteration.gettingEigenVector(matrix, eigenvalues[2])