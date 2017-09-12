import numpy as np
from numpy.linalg import norm

def hessenberg(A):
    m = len(A)
    for k in range(m-2):
        v = A[k+1:m, k]
        v[0] = v[0] + np.sign(v[0]) * norm(v)
        v = v / norm(v)
        P = np.eye(m-1-k) - 2.0 * (v * v.transpose())
        A[k+1:m, k:m] = P * A[k+1:m, k:m]
        A[0:m, k+1:m] = A[0:m, k+1:m] * P
    for i in range(m-2):
        for j in range(i+1):
            A[i+2, j] = 0
    return A

m = [[1.0, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
m = np.asmatrix(m)
m = hessenberg(m)
print(m)

#  function A = hessenbergform(A)
#     m = length(A);
#     for k=1:m - 2
#        v = A(k + 1:m, k);
#        v(1) = v(1) + sign(v(1)) * norm(v);
#        v = v / norm(v);
#        P = eye(m - k) - 2 * (v * v');
#        A(k + 1:m, k:m)=P * A(k + 1:m, k:m);
#        A(1:m, k + 1:m)=A(1:m, k + 1:m)*P;
#     end
#     for i=1:m - 2
#        for j=1:i
#           A(i + 2, j) = 0;
#        end
#     end
# end