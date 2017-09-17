from mna.tp01.utils.HouseHolder import *
from mna.tp01.tests.Runner import *
from mna.tp01.utils.Utils import *
from mna.tp01.utils.HouseHolder import *
import numpy as np
import unittest
from random import randint

_random_seed = randint(1,100000)
# _random_seed = 26221
# _random_seed = 26549
print("Testing with House Holder seed {0}.".format(_random_seed))


class HouseHolderTest4by4(unittest.TestCase):

    def test(self):
        matrix = mnaMat("2 0 2 3; 1 4 2 3; 2 5 6 3; 4 5 4 8")
        pyQ, pyR = np.linalg.qr(matrix)
        q, r = HouseHolder.qr(matrix)
        assertAbsEqualMatrix(q, pyQ)
        assertAbsEqualMatrix(r, pyR)


class HouseHolderTestSymetric(unittest.TestCase):
    def test(self):
        random = np.random
        random.seed(_random_seed)
        size = random.randint(100)
        matrix = random.rand(size, size) * 10000
        matrix = matrix.dot(matrix.T)
        (pyQ, pyR) = np.linalg.qr(matrix)
        (q, r) = HouseHolder.qr(matrix)
        assertAbsEqualMatrix(q, pyQ)
        assertAbsEqualMatrix(r, pyR, 4)


class HouseHolderTestNbyN(unittest.TestCase):

    def test(self):
        random = np.random
        random.seed(_random_seed)
        size = random.randint(100)
        matrix = random.rand(size, size)
        (pyQ, pyR) = np.linalg.qr(matrix)
        (q, r) = HouseHolder.qr(matrix)
        assertAbsEqualMatrix(q, pyQ)
        assertAbsEqualMatrix(r, pyR)


class HouseholderReflectorTest(unittest.TestCase):

        def test(self):
            vector = np.asarray([4, 3, 2, 6])
            reflector = HouseholderReflector(vector)
            result = np.dot(reflector, vector)
            print(reflector)
            print(result)
            assert result[0] == np.linalg.norm(vector)
