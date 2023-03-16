import numpy as np
import math
from scipy.linalg import sqrtm

# TODO: Implement a function for computing the inverse of a matrix (if it exists)   ---- Done! 
# TODO: Implement a function for computing the adjoint of a matrix

def MatrixAddition(matrix_a, matrix_b):
    matrix_a, matrix_b = np.array(matrix_a), np.array(matrix_b)
    m,n = matrix_a.shape
    matrix_c = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(m):
        for j in range(n):
            matrix_c[i][j] += matrix_a[i][j] + matrix_b[i][j]
    return np.array(matrix_c)

def MatrixSubtraction(matrix_a, matrix_b):
    matrix_a, matrix_b = np.array(matrix_a), np.array(matrix_b)
    m,n = matrix_a.shape
    matrix_c = [[0 for _ in range(m)] for _ in range(n)]
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

def innerProduct(matrix_a, matrix_b):
    matrix_a = np.array(matrix_a)
    matrix_b = np.array(matrix_b)
    a_t = matrix_a.transpose()
    if ((matrix_a.ndim == 1) and (matrix_b.ndim == 1)):
        if (matrix_a.shape[0] != matrix_b.shape[0]):
            print("Cannot take inner product of vectors that are of different shapes...")
            exit(0)
        inner_product = np.dot(matrix_a, matrix_b)
    else:
        inner_product = MatrixMultiplication(a_t, matrix_b)
    return inner_product

def outerProduct(matrix_a, matrix_b):
    matrix_a = np.array(matrix_a)
    matrix_b = np.array(matrix_b)
    b_t = matrix_b.transpose()
    if ((matrix_a.ndim == 1) and (matrix_b.ndim ==1)):
        matrix_a=matrix_a.reshape(-1,1)
        matrix_b=matrix_b.reshape(-1,1)
        matrix_b_transpose = matrix_b.transpose()
        outer_product = np.dot(matrix_a, matrix_b_transpose)
    else:
        outer_product = MatrixMultiplication(matrix_a, b_t)
    return outer_product

def compute_inverse(matrix):
    matrix = np.array(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        print("No unique inverse exists!", end=' ')
        pseudo_inverse = np.linalg.pinv(matrix)
        print("So calcluated the pseudo-inverse, which is:")
        return pseudo_inverse
    else:
        return np.linalg.inv(matrix)

def Frobenius_norm(matrix):
    matrix = np.array(matrix)
    sum_squares = 0.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            sum_squares += matrix[i][j]**2
    return math.sqrt(sum_squares)

def compute_trace(A):
    if len(A) != len(A[0]):
        return "The matrix is not square!"
    else:
        trace = 0.
        for i in range(len(A)):
            trace += A[i][i]
        return trace

def vector_norm(v, metric : int) -> float:
    norm = 0.
    for i in range(len(v)):
        norm += v[i]**metric
    return norm**(1/metric)

def matrix_norm(A, _norm):
    # Rather than this, you can also use my already defined function Frobenius_norm() above
    if(_norm == "Frobenius" or _norm == 'frobenius' or _norm == 'frob'):
        # return np.linalg.norm(A, ord='fro')  # Simply using the library function in NumPy
        norm = 0.
        for i in range(len(A)):
            for j in range(len(A[0])):
                norm += A[i][j]**2
        return np.sqrt(norm)
    elif (_norm == "Max" or _norm == 'max'):
        return np.max(np.abs(A))
    elif (_norm == "nuclear" or _norm == 'nuc' or _norm == 'Nuclear'):
        # return np.linalg.norm(A, ord='nuc')   # Simply using the library function in NumPy
        m = MatrixMultiplication(np.transpose(A), A)
        sqrt_m = sqrtm(m)
        return compute_trace(sqrt_m)