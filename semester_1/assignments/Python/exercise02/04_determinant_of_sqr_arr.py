'''
NumPy program to compute the determinant of a given square array
'''
import numpy as np

# create a square array
arr = np.array([[1, 2, 3], [3, 4, 5], [1, 2, 3]])

# compute the determinant
det = np.linalg.det(arr)

print("The determinant of the array is:", det)
