from mna.tp01.utils.QRAlgorithm import *
from mna.tp01.utils.Utils import *
from mna.tp01.tests.Runner import *
import numpy as np
import unittest
from random import randint

# _random_seed = randint(1,100000) #Set desired seed here
_random_seed = 54286 #Set desired seed here
print("Testing with QRAlgorithm seed {0}".format(_random_seed))


class TrivialQRAlgorithm3by3(unittest.TestCase):

    def test(self):
        matrix = mnaMat("1 0 3; 3 0 1; 5 8 4")
        matrix = matrix.dot(matrix.T)
        oVa,oVe = np.linalg.eig(matrix)
        Va,Ve = QRAlgorithm.trivialEig(matrix)
        assertAbsEqualMatrix(oVe,Ve)
        assertAbsEqualMatrix(np.sort(oVa),np.sort(Va))


class TrivialQRAlgorithmNbyN(unittest.TestCase):

    def test(self):
        random = np.random
        random.seed(_random_seed)
        size = random.randint(10,100)
        matrix = random.rand(size,size)
        matrix = matrix.dot(matrix.T)
        oVa,oVe = np.linalg.eig(matrix)
        Va,Ve = QRAlgorithm.trivialEig(matrix)
        assertAbsEqualMatrix(np.sort(oVa),np.sort(Va))
        assertAbsEqualMatrix(oVe,Ve)
