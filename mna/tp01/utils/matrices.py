import numpy as np


def GramSchmidtQR (matrix):
    n = len(matrix)
    q = np.zeros((n, n))
    r = np.zeros((n, n))
    for i in range(n):
        u = matrix[:, i]
        aux = u
        for j in range(i):
            aux = np.subtract(aux,(np.dot(u, q[:,j])*q[:,j]))
        u = aux
        e = u / np.linalg.norm(u)

        for j in range(i, n):
            r[i][j] = np.dot(matrix[:,j], e)

        for j in range(n):
            q[j][i] = e[j]
    return q, r





matrix = np.array([[1,2],[3, 4]])

Q,R = GramSchmidtQR(matrix)

print(Q)
print("//")
print(R)
print("//")
print(Q*R)
