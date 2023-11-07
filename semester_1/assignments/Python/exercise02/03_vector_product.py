'''
NumPy program to  compute  the  cross  product of two given vectors
'''
import numpy as np

# define the two vectors
vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])

# compute the cross product
cross_product = np.cross(vector1, vector2)

# print the result
print("The cross product of", vector1, "and", vector2, "is", cross_product)
