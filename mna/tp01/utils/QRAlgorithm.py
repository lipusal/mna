from mna.tp01.utils.GrandSchmidt import *
class QRAlgorithm:

    @staticmethod
    def QR (matrix, method=GrandSchmidt.QR):
        Q,R = method(matrix)
        eig_val = np.dot(np.transpose(Q),np.dot(Q,matrix))
        eig_vec = Q
        lastValue = matrix[0,0]
        while abs(eig_val[0,0]-lastValue) > 0.0001:
            lastValue = eig_val[0,0]
            Q,R = method(matrix)
            eig_val = np.dot(np.transpose(Q),np.dot(Q,eig_val))
            eig_vec = np.dot(eig_vec,Q)

        return eig_val, eig_vec



VA, VE = QRAlgorithm.QR(np.matrix("1 0; 1 4"))

print(VA)
print("----")
print(VE)
