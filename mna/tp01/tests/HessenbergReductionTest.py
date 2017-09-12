from mna.tp01.utils.QRAlgorithm import *
from mna.tp01.tests.Runner import *
import numpy as np
import scipy.linalg as sc
import unittest
from random import randint

_random_seed = randint(1,100000) #Set desired seed here
print("Testing with Hessenberg seed {0}".format(_random_seed))


class HessenbergTest4by4(unittest.TestCase):

    def test(self):
        matrix = np.matrix("2. 0 2 3; 1 4 2 3; 2 5 6 3; 4 5 4 8")
        oH = sc.hessenberg(matrix)
        H = QRAlgorithm.HessenbergReduction(matrix)
        assertEqualMatrix(H,oH)
        oVe,_ = np.linalg.eig(matrix)
        Ve,_ = np.linalg.eig(H)
        assertEqualMatrix(np.sort(oVe),np.sort(Ve))
        assertEqualMatrix(np.sort(np.linalg.eigvals(matrix)),np.sort(np.linalg.eigvals(H)))

class HessenbergTestSymetric(unittest.TestCase):

    def test(self):
        random = np.random
        random.seed(_random_seed)
        size = random.randint(10,100)
        matrix = random.rand(size,size)
        matrix = matrix.dot(matrix.T)
        oH = sc.hessenberg(matrix)
        H = QRAlgorithm.HessenbergReduction(matrix)
        assertEqualMatrix(H,oH)
        oVe,_ = np.linalg.eig(matrix)
        Ve,_ = np.linalg.eig(H)
        assertEqualMatrix(np.sort(oVe),np.sort(Ve))
        assertEqualMatrix(np.sort(np.linalg.eigvals(matrix)),np.sort(np.linalg.eigvals(H)))
        assertEqualMatrix(H,H.T)

class HessenbergTestNbyN(unittest.TestCase):

    def test(self):
        random = np.random
        random.seed(_random_seed)
        size = random.randint(10,100)
        matrix = random.rand(size,size)
        oH = sc.hessenberg(matrix)
        H = QRAlgorithm.HessenbergReduction(matrix)
        assertEqualMatrix(H,oH)
        oVe,_ = np.linalg.eig(matrix)
        Ve,_ = np.linalg.eig(H)
        assertEqualMatrix(np.sort(oVe),np.sort(Ve))
        assertEqualMatrix(np.sort(np.linalg.eigvals(matrix)),np.sort(np.linalg.eigvals(H)))
