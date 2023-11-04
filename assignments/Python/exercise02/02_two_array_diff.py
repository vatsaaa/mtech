'''
NumPy program to find the set difference of two arrays. 
Set difference returns the sorted, unique values in array1 that are not in array2.
'''
import numpy as np

# Define the two arrays
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([3, 4, 5, 6, 7])

# Find the set difference
result = np.setdiff1d(array1, array2)

# Print the result
print(result)
