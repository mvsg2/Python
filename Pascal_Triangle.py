# This is a program to print the Pascal's triangle till the row desired by the user

def printPascal(n):
  n = int(input("How many rows of the Pascal's triangle do you want to print? \n"))
  for i in range(1, n+1):
    for j in range(0, n-i+1):
      print(' ', end = ' ')
    
    C = 1
    for j in range(1, i + 1):
      print(' ', C, sep=' ', end=' ')
      
      C = C * (i -j) // j
    print()
    
#printPascal(6)
