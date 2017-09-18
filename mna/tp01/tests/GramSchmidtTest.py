from mna.tp01.utils.GramSchmidt import GramSchmidt
from mna.tp01.utils.Utils import *
from mna.tp01.tests.Runner import *
import numpy as np
import unittest
from random import randint

_random_seed = randint(1,100000) #Set desired seed here
print("Testing with GramSchmidt seed {0}".format(_random_seed))

class GramSchmidtTest3by3(unittest.TestCase):

    def test(self):
        matrix = mnaMat("1 0 3; 1 4 5; 2 5 7")
        oQ,oR = np.linalg.qr(matrix)
        Q,R = GramSchmidt.QR(matrix)
        assertAbsEqualMatrix(Q,oQ)
        assertAbsEqualMatrix(R,oR)
        assertEqualMatrix(matrix,np.dot(Q,R))

class GramSchmidtTesNbyN(unittest.TestCase):

    def test(self):
        random = np.random
        random.seed(_random_seed)
        size = random.randint(10,100)
        matrix = np.array(random.rand(size,size))
        oQ,oR = np.linalg.qr(matrix)
        Q,R = GramSchmidt.QR(matrix)
        assertAbsEqualMatrix(Q,oQ)
        assertAbsEqualMatrix(R,oR)
        assertEqualMatrix(matrix,Q.dot(R))
