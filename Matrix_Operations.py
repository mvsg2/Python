import numpy as np

def MatrixAddition(matrix_a, matrix_b):
    matrix_a, matrix_b = np.array(matrix_a), np.array(matrix_b)
    m,n = matrix_a.shape
    matrix_c = [[0 for i in range(m)] for j in range(n)]
    for i in range(m):
        for j in range(n):
            matrix_c[i][j] += matrix_a[i][j] + matrix_b[i][j]
    return np.array(matrix_c)

def MatrixSubtraction(matrix_a, matrix_b):
    matrix_a, matrix_b = np.array(matrix_a), np.array(matrix_b)
    m,n = matrix_a.shape
    matrix_c = [[0 for i in range(m)] for j in range(n)]
    for i in range(m):
        for j in range(n):
            matrix_c[i][j] += matrix_a[i][j] - matrix_b[i][j]
    return np.array(matrix_c)

def MatrixMultiplication(matrix_a, matrix_b):
    matrix_a, matrix_b = np.array(matrix_a), np.array(matrix_b)
    m,n = matrix_a.shape
    n,p= matrix_b.shape
    matrix_c = [[0 for j in range(p)] for i in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                matrix_c[i][j] += matrix_a[i][k]*matrix_b[k][j]
    return np.array(matrix_c)

def ifSymmetric(matrix):
    matrix = np.array(matrix)
    matrix_transpose = matrix.transpose()
    if ((matrix_transpose == matrix).sum() == matrix.shape[0]*matrix.shape[1]):
        print(f"The matrix:\n{matrix} is symmetric!")
    else:
        print(f"The matrix:\n{matrix} is not symmetric..")

def compute_determinant(matrix):
    matrix= np.array(matrix)
    if (matrix.shape[0] != matrix.shape[1]):
        print("Oops, the determinant of a non-square matrix cannot be computed..")
    det = np.linalg.det(matrix)
    print(f"The determinant of \n{matrix} is: %.3f"%det)