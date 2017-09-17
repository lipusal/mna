from mna.tp01.utils.QRAlgorithm import *
from mna.tp01.tests.Runner import *
from mna.tp01.utils.Utils import *
import numpy as np
import scipy.linalg as sc
import unittest
from random import randint

_random_seed = randint(1,100000) #Set desired seed here
print("Testing with Householder seed {0}".format(_random_seed))


class HouseholderTest4by4(unittest.TestCase):

    def test(self):
        vector = mnaMat("2. 0 2 3; 1 4 2 3; 2 5 6 3; 4 5 4 8")

