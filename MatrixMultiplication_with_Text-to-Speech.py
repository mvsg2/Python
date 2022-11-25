import numpy as np
from Audio_Utilities import *

s1 = "Defining a function for matrix-matrix multiplication..."
print(s1)
play_after_line(s1)
def MatrixMultiplication(matrix_a, matrix_b):
    m,n = matrix_a.shape
    n,p= matrix_b.shape
    matrix_c = [[0 for j in range(p)] for i in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                matrix_c[i][j] += matrix_a[i][k]*matrix_b[k][j]
    return np.array(matrix_c)
s2 = "Function for matrix-matrix multiplication defined!"
print(s2)
play_after_line(s2)

s3 = "Taking user inputs for the shapes of the matrices..."
print(s3)
play_after_line(s3)
m = int(input("Number of rows in the first matrix: "))
n = int(input("Number of rows in the second matrix: ")) # also equal to the no of cols in 1
p = int(input("Number of columns in the second matrix: "))

s4 = "Taking the elements of the first matrix from the user..."
print(s4)
play_after_line(s4)
A = np.zeros((m,n)).astype(int)
for i in range(m):
    for j in range(n):
        A[i][j] += int(input("A["+str(i)+"]["+str(j)+"]: "))

s5 = "Taking the elements of the second matrix from the user..."
print(s5)
play_after_line(s5)
B = np.zeros((n,p)).astype(int)
for i in range(n):
    for j in range(p):
        B[i][j] += int(input("B["+str(i)+"]["+str(j)+"]: "))

s6 = "Printing both the matrices..."
print(s6)
play_after_line(s6)
print("A is :\n", A)
print("B is :\n", B)

s7 = "Printing their product..."
print(s7)
play_after_line(s7)
C = MatrixMultiplication(A,B)
print("The product is: \n", C)
s8 = "The shape of their product is:"
print(s8)
play_after_line(s8)
print(C.shape)
