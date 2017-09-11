import unittest
import numpy as np

if __name__ == "__main__":
    all_tests = unittest.TestLoader().discover('.', pattern='*.py')
    unittest.TextTestRunner().run(all_tests)

def assertAbsEqualMatrix(M1,M2):
        np.testing.assert_array_almost_equal(np.asarray(np.abs(M1)),np.asarray(np.abs(M2)))

def assertEqualMatrix(M1,M2):
    np.testing.assert_array_almost_equal(np.asarray(M1),np.asarray(M2))
