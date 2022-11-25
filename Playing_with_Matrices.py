import numpy as np
# import time

"""
This program works (with both the functions being active) very well for square matrices.
For non-square matrices, be careful to check the shape compatibility of the matricess involved.
NOTE: This is a brute-force implementation. There are more efficient algorithms for operating on matrices.
"""

def MatrixAddition(matrix_a, matrix_b):
    m,n = matrix_a.shape
    matrix_c = [[0 for i in range(m)] for j in range(n)]
    for i in range(m):
        for j in range(n):
            matrix_c[i][j] += matrix_a[i][j] + matrix_b[i][j]
    return np.array(matrix_c)

def MatrixMultiplication(matrix_a, matrix_b):
    m,n = matrix_a.shape
    n,p= matrix_b.shape
    matrix_c = [[0 for i in range(m)] for j in range(n)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                matrix_c[i][j] += matrix_a[i][k]*matrix_b[k][j]
    return np.array(matrix_c)

a1 = int(input("Enter how many rows in the first matrix: "))
b1 = int(input("Enter how many columns in the first matrix: "))

a2 = int(input("Enter how many rows in the second matrix: "))
b2 = int(input("Enter how many columns in the second matrix: "))
print("\n")
first_matrix = [[0 for i in range(a1)] for j in range(b1)]
for i in range(a1):
    for j in range(b1):
        first_matrix[i][j] += int(input("first_matrix["+str(i)+"]["+str(j)+"]: "))
first_matrix = np.array(first_matrix)
print("\n")
second_matrix = [[0 for i in range(a2)] for j in range(b2)]
for i in range(a2):
    for j in range(b2):
        second_matrix[i][j] += int(input("second_matrix["+str(i)+"]["+str(j)+"]: "))
second_matrix = np.array(second_matrix)

print("\n")

print("The first matrix given is: \n", first_matrix)
print("The second matrix given is: \n", second_matrix)

# resultant_sum_matrix = MatrixAddition(first_matrix, second_matrix)
# print("The sum of the given matrices is:")
# time.sleep(0.5)
# print(resultant_sum_matrix)

resultant_product_matrix= MatrixMultiplication(first_matrix, second_matrix)
print("The product of the given matrices is: ")
# time.sleep(0.5)
print(resultant_product_matrix)