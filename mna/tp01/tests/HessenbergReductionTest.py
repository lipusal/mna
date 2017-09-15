from mna.tp01.utils.QRAlgorithm import *
from mna.tp01.tests.Runner import *
from mna.tp01.utils.Utils import *
import numpy as np
import scipy.linalg as sc
import unittest
from random import randint

_random_seed = randint(1,100000) #Set desired seed here
print("Testing with Hessenberg seed {0}".format(_random_seed))


class HessenbergTest4by4(unittest.TestCase):

    def test(self):
        matrix = mnaMat("2. 0 2 3; 1 4 2 3; 2 5 6 3; 4 5 4 8")
        oH = sc.hessenberg(matrix)
        H = QRAlgorithm.HessenbergReduction(matrix)
        assertEqualMatrix(H,oH)
        oVa,_ = np.linalg.eig(matrix)
        Va,_ = np.linalg.eig(H)
        assertAbsEqualMatrix(np.sort(oVa),np.sort(Va))

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
        oVa,_ = np.linalg.eig(matrix)
        Va,_ = np.linalg.eig(H)
        assertAbsEqualMatrix(np.sort(oVa),np.sort(Va))
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
        oVa,_ = np.linalg.eig(matrix)
        Va,_ = np.linalg.eig(H)
        assertAbsEqualMatrix(np.sort(oVa),np.sort(Va))
