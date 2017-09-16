import unittest
import numpy as np

if __name__ == "__main__":
    all_tests = unittest.TestLoader().discover('.', pattern='*.py')
    unittest.TextTestRunner().run(all_tests)

def assertAbsEqualMatrix(M1, M2, decimal=6):
    assertEqualMatrix(np.abs(M1),np.abs(M2), decimal)

def assertEqualMatrix(M1, M2, decimal=6):
    assert(M1.shape == M2.shape)
    for i in range(M1.shape[0]):
        np.testing.assert_array_almost_equal(np.asarray(M1[i]),np.asarray(M2[i]), decimal)
