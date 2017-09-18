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
        oVa,_ = np.linalg.eig(matrix)
        Va,_ = np.linalg.eig(H)
        assertAbsEqualMatrix(np.sort(oVa),np.sort(Va))
        assertEqualMatrix(H,oH)

class HessenbergTestSymetric(unittest.TestCase):

    def test(self):
        random = np.random
        random.seed(_random_seed)
        size = random.randint(10,100)
        matrix = random.rand(size,size)
        matrix = matrix.dot(matrix.T)
        oH = sc.hessenberg(matrix)
        H = QRAlgorithm.HessenbergReduction(matrix)
        oVa,_ = np.linalg.eig(matrix)
        Va,_ = np.linalg.eig(H)
        assertEqualMatrix(deepcopy(H),deepcopy(H.T))
        assertEqualMatrix(H,oH)
        assertAbsEqualMatrix(np.sort(oVa),np.sort(Va))

class HessenbergTestNbyN(unittest.TestCase):

    def test(self):
        random = np.random
        random.seed(_random_seed)
        size = random.randint(10,100)
        matrix = random.rand(size,size)
        oH = sc.hessenberg(matrix)
        H = QRAlgorithm.HessenbergReduction(matrix)
        oVa,_ = np.linalg.eig(matrix)
        Va,_ = np.linalg.eig(H)
        assertEqualMatrix(H,oH)
        assertAbsEqualMatrix(np.sort(oVa),np.sort(Va))


class HessenbergReflectorTest4by4(unittest.TestCase):

    def test(self):
        matrix = mnaMat("2. 0 2 3; 1 4 2 3; 2 5 6 3; 4 5 4 8")
        oH = sc.hessenberg(matrix)
        H, U = QRAlgorithm.HessenbergReductionWithReflector(matrix)
        oVa,oVe = np.linalg.eig(matrix)
        Va,Ve = np.linalg.eig(H)
        sVa,sVe = np.linalg.eig(oH)
        assertAbsEqualMatrix(H,oH)
        assertAbsEqualMatrix(np.sort(oVa),np.sort(Va))
        assertAbsEqualMatrix(oVe,U.dot(Ve).dot(np.linalg.inv(U)))

class HessenbergReflectorTestSymetric(unittest.TestCase):

    def test(self):
        random = np.random
        random.seed(_random_seed)
        size = random.randint(10,100)
        matrix = random.rand(size,size)
        matrix = matrix.dot(matrix.T)
        oH = sc.hessenberg(matrix)
        H = QRAlgorithm.HessenbergReductionWithReflector(matrix)
        oVa,_ = np.linalg.eig(matrix)
        Va,_ = np.linalg.eig(H)
        assertEqualMatrix(deepcopy(H),deepcopy(H.T))
        assertAbsEqualMatrix(np.sort(oVa),np.sort(Va))
        assertAbsEqualMatrix(H,oH)

class HessenbergReflectorTestNbyN(unittest.TestCase):

    def test(self):
        random = np.random
        random.seed(_random_seed)
        size = random.randint(10,100)
        matrix = random.rand(size,size)
        oH = sc.hessenberg(matrix)
        H = QRAlgorithm.HessenbergReductionWithReflector(matrix)
        oVa,_ = np.linalg.eig(matrix)
        Va,_ = np.linalg.eig(H)
        assertAbsEqualMatrix(np.sort(oVa),np.sort(Va))
        assertAbsEqualMatrix(H,oH)
