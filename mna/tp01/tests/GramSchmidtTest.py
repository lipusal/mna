from mna.tp01.utils.matrices import GrandSchmidt
import numpy as np
import unittest
from random import randint

_random_seed = 0
class GramchmidtTest3by3(unittest.TestCase):

    def test(self):
        random = np.random
        random.seed(_random_seed)
        matrix = random.rand(3,3)
        oQ,oR = np.abs(np.linalg.qr(matrix))
        Q,R = np.abs(GrandSchmidt.QR(matrix))
        np.testing.assert_array_almost_equal(np.asarray(Q),np.asarray(oQ))
        np.testing.assert_array_almost_equal(np.asarray(R),np.asarray(oR))

class GramchmidtTesNbyN(unittest.TestCase):

    def test(self):
        random = np.random
        random.seed(_random_seed)
        size = random.randint(100)
        matrix = random.rand(size,size)
        oQ,oR = np.abs(np.linalg.qr(matrix))
        Q,R = np.abs(GrandSchmidt.QR(matrix))
        np.testing.assert_array_almost_equal(np.asarray(Q),np.asarray(oQ))
        np.testing.assert_array_almost_equal(np.asarray(R),np.asarray(oR))



if __name__ == '__main__':
    # _random_seed = randint(1,100000)
    _random_seed = 25
    print("Testing with seed {0}".format(_random_seed))
    unittest.main()
