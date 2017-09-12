import numpy as np

def inner(M1,M2):
    return np.dot(row(M1),col(M2))

def col(V):
    return V.reshape((V.size, 1))

def row(V):
    return V.reshape((1,V.size))
