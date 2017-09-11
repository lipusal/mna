from mna.tp01.utils.matrices import GrandSchmidt
from mna.tp01.tests.Runner import *
import numpy as np
import unittest
from random import randint

_random_seed = randint(1,100000)
print("Testing with GrandSchmidt seed {0}".format(_random_seed))

class GrandSchmidtTest3by3(unittest.TestCase):

    def test(self):
        random = np.random
        random.seed(_random_seed)
        matrix = random.rand(3,3)
        oQ,oR = np.linalg.qr(matrix)
        Q,R = GrandSchmidt.QR(matrix)
        assertAbsEqualMatrix(Q,oQ)
        assertAbsEqualMatrix(R,oR)
        assertEqualMatrix(matrix,np.dot(Q,R))

class GrandSchmidtTesNbyN(unittest.TestCase):

    def test(self):
        random = np.random
        random.seed(_random_seed)
        size = random.randint(100)
        matrix = random.rand(size,size)
        oQ,oR = np.linalg.qr(matrix)
        Q,R = GrandSchmidt.QR(matrix)
        assertAbsEqualMatrix(Q,oQ)
        assertAbsEqualMatrix(R,oR)
        assertEqualMatrix(matrix,np.dot(Q,R))
