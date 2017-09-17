from mna.tp01.utils.Utils import *
from mna.tp01.utils.InverseIteration import InverseIteration
from mna.tp01.tests.Runner import *
import numpy as np
import unittest
from random import randint

_random_seed = randint(1,100000) #Set desired seed here
print("Testing with InverseIteration seed {0}".format(_random_seed))

#
# class InverseIterationTest2by2(unittest.TestCase):
#
#     def test(self):
#         matrix = np.random.rand(2, 2)
#         while np.linalg.det(matrix) == 0:
#             _random_seed = randint(1, 100000)  # Set desired seed here
#             matrix = np.random.rand(2, 2)
#         eigenvalues,oeigenvectors = np.linalg.eig(matrix)
#         eigenvectors = np.zeros((2,2))
#         # print(eigenvectors[0,:])
#         # print("\noutput Eigenvectors:")
#         # print (oeigenvectors[1,:])
#         for i in range(len(eigenvalues)):
#             aux = InverseIteration.gettingEigenVector(matrix,eigenvalues[i])
#             for j in range(len(aux)):
#                 eigenvectors[j][i] = aux[j][0]
#         print("\nExpected EigenVectors:")
#         print (oeigenvectors)
#         print("\nActual EigenVectors:")
#         print (eigenvectors)
#         print("\nerrorMatrix:")
#         print(np.subtract(np.absolute(oeigenvectors),np.absolute(eigenvectors)))
#         assertAlmostEqualMatrix(eigenvectors,oeigenvectors)

class InverseIterationTesNbyN(unittest.TestCase):

    def test(self):
        size = np.random.randint(10, 100)
        matrix = np.array(np.random.rand(size, size))
        # while np.linalg.det(matrix) == 0:
        #     _random_seed = randint(1, 100000)  # Set desired seed here
        #     matrix = np.random.rand(size, size)
        matrix = np.dot(matrix.T,matrix) #to guarantee real eigenvalues
        eigenvalues,oeigenvectors = np.linalg.eig(matrix)
        eigenvectors = np.zeros((size,size))
        for i in range(len(eigenvalues)):
            aux = InverseIteration.gettingEigenVector(matrix,eigenvalues[i])
            for j in range(len(aux)):
                eigenvectors[j][i] = aux[j][0]

        #seeds 31921  67766
        # print("\nExpected EigenVectors:")
        # print (oeigenvectors)
        # print("\nActual EigenVectors:")
        # print (eigenvectors)
        # print("\nerrorMatrix:")
        #print(np.subtract(np.absolute(oeigenvectors),np.absolute(eigenvectors)))
        assertAbsEqualMatrix(eigenvectors,oeigenvectors)



