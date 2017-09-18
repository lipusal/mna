from pytest_benchmark.plugin import benchmark

from mna.tp01.utils.QRAlgorithm import *
from mna.tp01.tests.Runner import *
from mna.tp01.utils.Utils import *
import numpy as np
import scipy.linalg as sc
import unittest
import pytest_benchmark

from random import randint

_random_seed = randint(1,100000) #Set desired seed here
print("Testing with Hessenberg seed {0}".format(_random_seed))


class HessenbergBasicTest4by4(unittest.TestCase):

    def test(self):
        matrix = mnaMat("2. 0 2 3; 1 4 2 3; 2 5  6 3; 4 5 4 8")
        oH = sc.hessenberg(matrix)
        H = QRAlgorithm.HessenbergReduction(matrix)
        oVa,_ = np.linalg.eig(matrix)
        Va,_ = np.linalg.eig(H)
        assertAbsEqualMatrix(np.sort(oVa),np.sort(Va))
        assertEqualMatrix(H,oH)

class HessenbergBasicTestSymetric(unittest.TestCase):

    def test(self):
        random = np.random
        random.seed(_random_seed)
        size = random.randint(10,20)
        matrix = random.rand(size,size)
        matrix = matrix.dot(matrix.T)
        oH = sc.hessenberg(matrix)
        H = QRAlgorithm.HessenbergReduction(matrix)
        oVa,_ = np.linalg.eig(matrix)
        Va,_ = np.linalg.eig(H)
        assertEqualMatrix(deepcopy(H),deepcopy(H.T))
        assertEqualMatrix(H,oH)
        assertAbsEqualMatrix(np.sort(oVa),np.sort(Va))

class HessenbergBasicTestNbyN(unittest.TestCase):

    def test(self):
        random = np.random
        random.seed(_random_seed)
        size = random.randint(10,50)
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
        assertAbsEqualMatrix(H,oH)
        assertAbsEqualMatrix(np.sort(oVa),np.sort(Va))
        assertAbsEqualMatrix(oVe,U.dot(Ve))

class HessenbergReflectorTestSymetric(unittest.TestCase):

    def test(self):
        random = np.random
        random.seed(_random_seed)
        size = random.randint(10,20)
        matrix = random.rand(size,size)
        matrix = matrix.dot(matrix.T)
        oH = sc.hessenberg(matrix)
        H, U = QRAlgorithm.HessenbergReductionWithReflector(matrix)
        oVa,oVe = np.linalg.eig(matrix)
        Va,Ve = np.linalg.eig(H)
        assertEqualMatrix(deepcopy(H),deepcopy(H.T))
        assertAbsEqualMatrix(np.sort(oVa),np.sort(Va))
        assertAbsEqualMatrix(H,oH)
        assertAbsEqualMatrix(oVe,U.dot(Ve))

class HessenbergReflectorTestNbyN(unittest.TestCase):

    def test(self):
        random = np.random
        random.seed(_random_seed)
        size = random.randint(10,50)
        matrix = random.rand(size,size)
        oH = sc.hessenberg(matrix)
        H, U = QRAlgorithm.HessenbergReductionWithReflector(matrix)
        oVa,oVe = np.linalg.eig(matrix)
        Va,Ve = np.linalg.eig(H)
        assertAbsEqualMatrix(np.sort(oVa),np.sort(Va))
        assertAbsEqualMatrix(H,oH)
        assertAbsEqualMatrix(oVe,U.dot(Ve))
