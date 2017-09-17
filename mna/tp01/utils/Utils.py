import numpy as np

def inner(M1,M2):
    return np.dot(row(M1),col(M2))

def col(V):
    return V.reshape((V.size, 1))

def row(V):
    return V.reshape((1,V.size))

def mnaMat(data):
    return np.array(np.matrix(data))

def sortMatrix(matrix):
    np.matrix.sort(matrix, axis=0)
    if np.size(matrix[0])>1:
        np.matrix.sort(matrix,axis=1)
    return matrix
